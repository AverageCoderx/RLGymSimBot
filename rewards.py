import numpy as np # Import numpy, the python math library
import math
from numpy.linalg import norm
from math import exp
from rlgym_sim.utils.math import cosine_similarity
from rlgym_sim.utils import RewardFunction # Import the base RewardFunction class
from rlgym_sim.utils.gamestates import GameState, PlayerData # Import game state stuff
from rlgym_sim.utils.common_values import (BLUE_GOAL_BACK, BLUE_GOAL_CENTER, ORANGE_GOAL_BACK,
                                       ORANGE_GOAL_CENTER, CAR_MAX_SPEED, ORANGE_TEAM, CAR_MAX_SPEED, BALL_MAX_SPEED, ORANGE_TEAM, BLUE_TEAM,)

def distance2D(p1, p2):
    return np.linalg.norm(p2 - p1)

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, norm_vel))

class AlignBallGoal(RewardFunction):
    def __init__(self, defense=1., offense=1.):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc

        # Align player->ball and net->player vectors
        defensive_reward = self.defense * math.cosine_similarity(ball - pos, pos - protecc)

        # Align player->ball and player->net vectors
        offensive_reward = self.offense * math.cosine_similarity(ball - pos, attacc - pos)

        return defensive_reward + offensive_reward

class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:

        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array

        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0

class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity

        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)

        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)

        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)

        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0

class AirTouchReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        MAX_TIME_IN_AIR = 1.75 # A rough estimate of the maximum reasonable aerial time
        reward = 0
        if self.ball_touched:
            air_time_frac = min(player.air_time, MAX_TIME_IN_AIR) / MAX_TIME_IN_AIR
            height_frac = ball.position[2] / CommonValues.CEILING_Z
            reward = min(air_time_frac, height_frac)
        return reward

class BallTouchReward(RewardFunction):
    # Default constructor
    def __init__(self, Beginner: bool):
        super().__init__()
        self.isBeginner = Beginner
        self.ballVelLast = 0  # Initialize the previous ball velocity to 0

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        self.ballVelLast = 0  # Reset ball velocity on game reset

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0

        if self.isBeginner:
            if player.ball_touched:  # Use player data for ball touch check
                reward += 1.0
        else:
            if player.ball_touched:
                # Calculate velocity gain
                current_ball_velocity = np.linalg.norm(state.ball.linear_velocity)  # Magnitude of ball velocity
                velocity_gain = current_ball_velocity - self.ballVelLast
                reward += max(0, velocity_gain)  # Reward only positive velocity gain
                self.ballVelLast = current_ball_velocity  # Update ball velocity

        return reward

class FaceBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff))

class LandingRecoveryReward(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.up = np.array([0, 0, 1])

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and player.car_data.linear_velocity[2] < 0
            and player.car_data.position[2] > 250
        ):
            flattened_vel = normalize(
                np.array(
                    [
                        player.car_data.linear_velocity[0],
                        player.car_data.linear_velocity[1],
                        0,
                    ]
                )
            )
            forward = player.car_data.forward()
            flattened_forward = normalize(np.array([forward[0], forward[1], 0]))
            reward += flattened_vel.dot(flattened_forward)
            reward += self.up.dot(player.car_data.up())
            reward /= 2

        return reward

class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        car_speed = np.linalg.norm(player.car_data.linear_velocity)
        car_dir = sign(player.car_data.forward().dot(player.car_data.linear_velocity))
        if car_dir < 0:
            car_speed /= -2300

        else:
            car_speed /= 2300
        return min(car_speed, 1)

class AerialNavigation(RewardFunction):
    # TODO
    # Make it reward for multiple touches in the air
    # Make it reward for being closer to the ball
    def __init__(
        self, ball_height_min=400, player_height_min=200, beginner=True  # make sure to change beginner eventually
    ) -> None:
        super().__init__()
        self.ball_height_min = ball_height_min
        self.player_height_min = player_height_min
        self.face_reward = FaceBallReward()
        self.beginner = beginner
        self.previous_distance = None

    def reset(self, initial_state: GameState) -> None:
        self.face_reward.reset(initial_state)
        self.previous_distance = None

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0

        if (
            not player.on_ground
            and state.ball.position[2] > self.ball_height_min
            > player.car_data.position[2]
            and player.car_data.linear_velocity[2] > 0
            and distance2D(player.car_data.position, state.ball.position)
            < state.ball.position[2] * 3
        ):
            # Velocity check: alignment between player and ball
            ball_direction = normalize(state.ball.position - player.car_data.position)
            alignment = ball_direction.dot(normalize(player.car_data.linear_velocity))

            # Reward alignment
            if self.beginner:
                reward += max(0, alignment * 0.5)

            reward += alignment * (np.linalg.norm(player.car_data.linear_velocity) / 2300.0)

            # Reward for getting closer to the ball
            current_distance = distance2D(player.car_data.position, state.ball.position)
            if self.previous_distance is not None:
                distance_diff = self.previous_distance - current_distance
                if distance_diff > 0:
                    reward += 0.5
            self.previous_distance = current_distance

            if player.ball_touched:
                reward = reward * 1.5
        return max(reward, 0)

class BoostReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return math.sqrt(player.boost_amount * 100) / 10 #do this if it returns between 0 - 100

class BoostPickupReward(RewardFunction):
    # Constructor to initialize prevBoost
    def __init__(self):
        super().__init__()
        self.prevBoost = 0  # Store previous boost amount as an instance variable

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        self.prevBoost = 0  # Reset previous boost amount on reset

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0

        # If the player's boost has increased, reward the difference
        if player.boost_amount > self.prevBoost:
            reward = player.boost_amount - self.prevBoost

        # Update prevBoost for the next step
        self.prevBoost = player.boost_amount
        return reward # / 100 do this if it returns 0 - 100

# start of necto rewards seperated by chatgpt, we'll see how this goes...

class GoalSpeedBonusReward(RewardFunction):
    def __init__(self, goal_speed_bonus_w=1.0):
        super().__init__()
        self.goal_speed_bonus_w = goal_speed_bonus_w
        self.last_ball_velocity = np.zeros(3)
        self.last_blue_score = 0
        self.last_orange_score = 0

    def reset(self, initial_state: GameState):
        self.last_ball_velocity = initial_state.ball.linear_velocity
        self.last_blue_score = initial_state.blue_score
        self.last_orange_score = initial_state.orange_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        d_blue = state.blue_score - self.last_blue_score
        d_orange = state.orange_score - self.last_orange_score
        ball_velocity = state.ball.linear_velocity
        self.last_blue_score = state.blue_score
        self.last_orange_score = state.orange_score

        reward = 0.0
        if d_blue > 0 or d_orange > 0:
            goal_speed = 0.0
            if d_blue > 0:
                goal_speed = d_blue * norm(self.last_ball_velocity)
            elif d_orange > 0:
                goal_speed = d_orange * norm(self.last_ball_velocity)

            bonus = self.goal_speed_bonus_w * (goal_speed / BALL_MAX_SPEED)
            if player.team_num == ORANGE_TEAM and d_orange > 0:
                reward += bonus
            elif player.team_num != ORANGE_TEAM and d_blue > 0:
                reward += bonus
            elif player.team_num == ORANGE_TEAM and d_blue > 0:
                reward -= bonus
            elif player.team_num != ORANGE_TEAM and d_orange > 0:
                reward -= bonus

        self.last_ball_velocity = ball_velocity
        return float(reward)

class AlignmentReward(RewardFunction):
    def __init__(self, align_w=1.0):
        super().__init__()
        self.align_w = align_w

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_pos = state.ball.position
        player_pos = player.car_data.position

        if player.team_num == ORANGE_TEAM:
            goal_vector = np.array(ORANGE_GOAL_BACK) - player_pos
        else:
            goal_vector = np.array(BLUE_GOAL_BACK) - player_pos

        player_to_ball = ball_pos - player_pos
        alignment = cosine_similarity(player_to_ball, goal_vector)

        reward = self.align_w * alignment
        return float(reward)

class BoostLoseReward(RewardFunction):
    def __init__(self, boost_lose_w=1.0):
        super().__init__()
        self.boost_lose_w = boost_lose_w
        self.last_boost_amount = {}

    def reset(self, initial_state: GameState):
        self.last_boost_amount = {
            player.car_id: player.boost_amount
            for player in initial_state.players
        }

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        last_boost = self.last_boost_amount.get(player.car_id, player.boost_amount)
        current_boost = player.boost_amount
        boost_diff = np.sqrt(np.clip(current_boost, 0, 1)) - np.sqrt(np.clip(last_boost, 0, 1))
        self.last_boost_amount[player.car_id] = current_boost

        reward = 0.0
        if boost_diff < 0:
            car_height = player.car_data.position[2]
            penalty = self.boost_lose_w * boost_diff * (1 - car_height / 642.775) #goal height
            reward += penalty
        return float(reward)

class FlipResetReward(RewardFunction):
    def __init__(self, heightScaling=0.25, minimumHeight = 100):
        super().__init__()
        self.heightScaling = heightScaling
        self.minimumHeight = minimumHeight
        self.prevFlip = {}

    def reset(self, initial_state: GameState):
        # Initialize previous flip status for all players
        self.prevFlip = player.has_flip

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0

        # Check if the player has just gained a flip and is above a certain height
        if not self.prevFlip and player.has_flip and player.car_data.position[2] > self.minimumHeight:
            reward += player.car_data.position[2] * self.heightScaling

        # Update the previous flip status for the player
        self.prevFlip = player.has_flip

        return reward

# start of opti's rewards
class FlipResetRewardOPTI(RewardFunction):
    def __init__(self):
        super().__init__()
        # Reward weights and parameters (set these based on your needs)
        self.flip_reset_w = 10  # Reward for a successful flip reset
        self.quick_flip_reset_w = 5  # Reward for quick first flip reset
        self.quick_flip_reset_norm_steps = 100  # Normalization steps for quick flip reset
        self.flip_reset_delay_steps = 50  # Delay steps for subsequent resets
        self.inc_flip_reset_w = 2  # Incremental reward for consecutive resets
        self.prevent_chain_reset = True  # Whether to prevent consecutive resets
        self.cancel_flip_reset_indices = None  # Actions to cancel a flip reset (can be filled with a list of indices)

        self.got_reset = [False] * 8  # Keep track of reset status for each player (assuming 8 players)
        self.cons_resets = 0  # Counter for consecutive resets
        self.reset_timer = -100000  # Timer to track time since last reset
        self.kickoff_timer = 1000  # Set the time until the kickoff is finished

    def reset(self, initial_state: GameState):
        # Reset logic when the game resets (this is called at the start of the game)
        self.got_reset = [False] * 8  # Reset reset status for each player
        self.cons_resets = 0  # Reset the consecutive reset counter
        self.reset_timer = -100000  # Reset the reset timer

    def get_reward(self, player: PlayerData, state: GameState) -> float:
        reward = 0

        # Loop over all players (assuming `player` is the current player whose reward is being calculated)
        i = player.index  # Get the player's index
        last = state.players[i - 1]  # The last player's data (for checking the jump state)

        # Check for flip reset (first flip reset of the episode)
        if not last.has_jump and player.has_jump and state.ball.position[2] > 200 and \
                np.linalg.norm(state.ball.position - player.car_data.position) < 110 and \
                cosine_similarity(state.ball.position - player.car_data.position, -player.car_data.up()) > 0.9:
            if not self.got_reset[i]:  # First reset of episode
                reward += self.quick_flip_reset_w * self.quick_flip_reset_norm_steps / self.kickoff_timer
            self.got_reset[i] = True

            # Reward for successful flip reset (after first reset)
            if (self.kickoff_timer - self.reset_timer > self.flip_reset_delay_steps and self.prevent_chain_reset) or \
                    not self.prevent_chain_reset:
                if previous_action is not None and self.cancel_flip_reset_indices is not None and \
                        previous_action[i] not in self.cancel_flip_reset_indices:
                    reward += self.flip_reset_w
                self.cons_resets += 1
                if self.cons_resets > 1:
                    reward += self.inc_flip_reset_w * min((1.4 ** self.cons_resets), 6) / 6
            self.reset_timer = self.kickoff_timer

        # Reset the counters if the player is on the ground (no more flip resets allowed)
        elif player.on_ground:
            self.cons_resets = 0
            self.reset_timer = -100000

        return reward

class FlipResetHelperRewardOPTI(RewardFunction):
    def __init__(self, flip_reset_help_w=1.0):
        """
        Constructor to initialize the flip reset help weight.
        :param flip_reset_help_w: Weight to scale the flip reset reward (default is 1.0).
        """
        super().__init__()
        self.flip_reset_help_w = flip_reset_help_w

    def reset(self, initial_state: GameState):
        """
        Reset does nothing in this case.
        """
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Calculate the reward for a specific player, at the current state.
        This reward encourages the bot to position itself for a potential flip reset.

        :param player: The PlayerData object representing the player.
        :param state: The current GameState.
        :param previous_action: The previous action taken by the player.
        :return: The calculated reward.
        """
        reward = 0

        if self.flip_reset_help_w != 0:
            # Calculate the 'upness' of the player's car (how aligned it is with the ceiling)
            upness = cosine_similarity(
                np.asarray([0, 0, CEILING_Z - player.car_data.position[2]]),
                -player.car_data.up()  # bottom of the car points towards the ceiling
            )

            # Calculate how far the player is from the walls
            from_wall_ratio = min(1, abs(state.ball.position[0]) / 1300)

            # Calculate the height of the ball relative to the field
            height_ratio = min(1, state.ball.position[2] / 1700)

            # Calculate how aligned the ball is with the bottom of the car
            bottom_ball_ratio = 2 * cosine_similarity(
                state.ball.position - player.car_data.position, -player.car_data.up()
            )

            # Determine the goal objective based on the player's team
            if player.team_num == BLUE_TEAM:
                objective = np.array(ORANGE_GOAL_BACK)
            else:
                objective = np.array(BLUE_GOAL_BACK)

            # Calculate how aligned the player is with the objective (goal)
            align_ratio = cosine_similarity(
                objective - player.car_data.position, player.car_data.forward()
            )

            # Calculate the positional difference between the player and the ball, with extra weight on the Z-axis (height)
            pos_diff = state.ball.position - player.car_data.position
            pos_diff[2] *= 2  # Make the Z-axis difference more important
            norm_pos_diff = np.linalg.norm(pos_diff)

            # Calculate the final flip reset reward based on the factors
            flip_rew = bottom_ball_ratio * from_wall_ratio * height_ratio * align_ratio * \
                       np.clip(-1, 1, 40 * upness / (norm_pos_diff + 1))

            # Apply the weight factor to the final reward
            reward += self.flip_reset_help_w * flip_rew

        return reward

class AerialDistanceReward(RewardFunction):
    def __init__(self, height_scale = 0.5, distance_scale = 0.5):
        """
        Initializes the AerialDistanceReward function.
        :param height_scale: Scaling factor for rewarding height during aerial play.
        :param distance_scale: Scaling factor for rewarding distances traveled during aerial play.
        """
        super().__init__()
        self.height_scale = height_scale  # Scale for height-based rewards
        self.distance_scale = distance_scale  # Scale for distance-based rewards

        # Variables to track current player, previous state, and distances
        self.current_car: Optional[PlayerData] = None
        self.prev_state: Optional[GameState] = None
        self.ball_distance: float = 0  # Distance the ball has traveled
        self.car_distance: float = 0  # Distance the car has traveled

    def reset(self, initial_state: GameState):
        """
        Resets the state of the reward function.
        :param initial_state: The initial state of the game.
        """
        self.current_car = None  # No player currently tracked
        self.prev_state = initial_state  # Store the initial state for comparison

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Calculates the aerial distance reward for the given player.
        :param player: The current player.
        :param state: The current game state.
        :param previous_action: The last action taken by the player (not used here).
        :return: A scaled reward value for aerial distance and touches.
        """
        rew = 0  # Initialize reward
        is_current = self.current_car is not None and self.current_car.car_id == player.car_id  # Check if player is tracked

        # Check if the player is on the ground
        if player.car_data.position[2] < RAMP_HEIGHT:
            if is_current:  # If the tracked player is now on the ground, reset tracking
                is_current = False
                self.current_car = None
        # Detect the first aerial touch
        elif player.ball_touched and not is_current:
            is_current = True  # Start tracking the player
            self.ball_distance = 0  # Reset ball distance
            self.car_distance = 0  # Reset car distance
            # Reward for initial aerial height, scaled by height_scale
            rew = self.height_scale * max(player.car_data.position[2] + state.ball.position[2] - 2 * RAMP_HEIGHT, 0)
        # If the player is still in the air after the initial touch
        elif is_current:
            # Accumulate car travel distance since the last frame
            self.car_distance += np.linalg.norm(player.car_data.position - self.current_car.car_data.position)
            # Accumulate ball travel distance since the last frame
            self.ball_distance += np.linalg.norm(state.ball.position - self.prev_state.ball.position)
            # Reward for additional touches, based on accumulated distances
            if player.ball_touched:
                rew = self.distance_scale * (self.car_distance + self.ball_distance)  # Reward based on total distance
                self.car_distance = 0  # Reset car distance
                self.ball_distance = 0  # Reset ball distance

        if is_current:
            # Update current car to the latest player data for tracking
            self.current_car = player

        # Update previous state for the next frame comparison
        self.prev_state = state

        # Normalize reward by the maximum possible distance (2 * BACK_WALL_Y)
        return rew / (2 * BACK_WALL_Y)

class ExampleReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        #reward logic here
        return reward

