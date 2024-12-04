import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_sim.utils.gamestates.player_data import PlayerData
from rlgym_ppo.util import MetricsLogger
import rewards
from rewards import InAirReward, SpeedTowardBallReward, BallTouchReward, FaceBallReward, LandingRecoveryReward, SpeedReward, AerialNavigation, BoostReward, BoostPickupReward, FlipResetReward, AlignmentReward, BoostLoseReward, GoalSpeedBonusReward, AerialDistanceReward
from zerosumreward import ZeroSumReward

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [
                game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score,  # Orange team's score
                game_state.players[0].boost_amount,  # Player's boost amount (fixed reference)
                game_state.players[0].on_ground,  # Whether the player is on the ground
                game_state.players[0].match_goals,  # Goals scored by the player
                game_state.players[0].match_demolishes,  # Demolitions caused by the player
                game_state.players[0].ball_touched,
                game_state.ball.linear_velocity  # Ball's linear velocity
               ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_car_vel = 0  # Initialize avg_car_vel
        avg_ball_vel = 0  # Initialize avg_ball_vel
        avg_goals = 0
        avg_boost = 0
        avg_airtime = 0
        avg_demos = 0
        avg_touches = 0

        for metric_array in collected_metrics:
            car_velocity = metric_array[0]
            car_vel_magnitude = np.linalg.norm(car_velocity)  # Calculate the magnitude of the car's velocity
            avg_car_vel += car_vel_magnitude  # Accumulate car velocity magnitude

            avg_boost += metric_array[3]
            avg_airtime += 1 - metric_array[4]  # Time spent in the air
            avg_goals += metric_array[5]
            avg_demos += metric_array[6]
            avg_touches += metric_array[7]

            ball_velocity = metric_array[8]
            ball_vel_magnitude = np.linalg.norm(ball_velocity)  # Calculate the magnitude of the ball's velocity
            avg_ball_vel += ball_vel_magnitude  # Accumulate ball velocity magnitude

        avg_car_vel /= len(collected_metrics)
        avg_ball_vel /= len(collected_metrics)
        avg_boost /= len(collected_metrics)
        avg_airtime /= len(collected_metrics)
        avg_goals /= len(collected_metrics)
        avg_demos /= len(collected_metrics)
        avg_touches /= len(collected_metrics)

        report = {
            "average player speed": avg_car_vel,
            "average ball speed": avg_ball_vel,
            "average boost": avg_boost,
            "average airtime": avg_airtime,
            "average goals": avg_goals,
            "average demos": avg_demos,
            "ball touch ratio": avg_touches,
            "Cumulative Timesteps": cumulative_timesteps
        }

        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
        EventReward
    from lookupact import LookupAction
    from customobs import AdvancedObsPadder
    from rlgym_sim.utils.state_setters.random_state import RandomState
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    obs_builder = AdvancedObsPadder(2)
    state_setter = RandomState(True, True, False)

    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    # zero sum reward usage  ZeroSumReward(YourReward(), team_spirit=whatever), yourweight)
    # make demo reward = goal reward for it to learn demos
    reward_fn = CombinedReward.from_zipped( # Format is (func, weight)
        (InAirReward(), 0.15),
        (FaceBallReward(), 0.1),
        (SpeedTowardBallReward(), 1.0),
        (VelocityBallToGoalReward(), 2.0),
        (BoostReward(), 0.5),
        (BoostPickupReward(), 1.0),
        (BallTouchReward(False), 0.5), #true meaning it is a beginner. if true, touch returns full reward, if false, touch is scaled with velocity gained
        (EventReward(team_goal=1, concede=-1, demo=0.1), 40.0)
    )


    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)

    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # 16 processes
    n_proc = 24
    policy_layer_sizes = (512, 1024, 512, 256)
    critic_layer_sizes = (1024, 1024, 512, 256)

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
                      build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      policy_layer_sizes=policy_layer_sizes,
                      critic_layer_sizes=critic_layer_sizes,
                      ppo_batch_size=50000, #50k default
                      ts_per_iteration=50000, #50k default
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.001,
                      ppo_epochs=2,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=5_000_000,
                      timestep_limit=1_000_000_000,
                      log_to_wandb=True,
                      render = False, # if rendering, change n_proc to 1
                      render_delay = 0.03
                      )
    learner.learn()
