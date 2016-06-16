import pygame, sys, random, pickle, math
from pygame.locals import *

# Colors
#             R    G    B
BROWN     = ( 82,  70,  70)
GREEN     = ( 74, 140,  74)
BLUE      = ( 93,  71, 173)
PEACH     = (242, 146,  82)
WHITE     = (255, 255, 255)

PLAYER_COLOR = PEACH
PROJECTILE_COLORS = (GREEN, BLUE)

# Global Conatants
WINDOW_WIDTH  = 1400 # Window height
WINDOW_HEIGHT = 800 # Window width
FPS = 40 # Frames per second
NUMBER_OF_PROJECTILES = 45 # Number of projectiles on the screen

def main():
    """ Main program function. """

    # initialise pygame
    pygame.init()
    # create the window
    screen_dimensions = ((WINDOW_WIDTH, WINDOW_HEIGHT))
    screen = pygame.display.set_mode(screen_dimensions)
    pygame.display.set_caption('Collision Game')

    # initialise the fps clock to regulate the fps
    fps_clock = pygame.time.Clock()

    # create an instance of the Game() class
    game = Game()

    while True:
        # process events eg keystrokes etc.
        game.process_events()

        # run the game logic and check for collisions
        game.logic()

        # render the player and projectiles or the game over screen
        game.render(screen)

        # regulate the game fps
        fps_clock.tick(FPS)

class Game(object):
    """ This class represents the game. """

    # -- Game Attributes --
    # 
    projectile_list = None
    all_sprites_list = None
    player = None
    pygame.font.init()
    game_over = False

    # the time between score updates
    time_to_score_update = 1000 # milliseconds
    # holds the score
    score = 0
    # initialise the clock to keep track of the time between score updates
    score_clock = pygame.time.Clock()
    # font that the current score will be displayed in
    score_font = pygame.font.Font("freesansbold.ttf", 30)
    # highest score that the player got scince they opened the game
    high_score = 0

    # time between projectiles changing direction
    time_between_direction_switches = 2000 # milliseconds
    # initialise the clock to keep track of the time between projectile direction updates
    projectile_direction_clock = pygame.time.Clock()

    # width, height and speed of the projectiles and player
    width = 20
    height = 20

    player_speed = 4

    # variables to hold the updated speed before it is updated with player.update_movement()
    player_x_movement = player_speed
    player_y_movement = 0

    def __init__(self):
        self.game_over = False

        self.score = 0
        self.time_since_last_tick = 0
        self.score_clock.tick()

        # set the time scince the lat projectile switch to 0
        self.time_scince_last_direction_switch = 0
        self.projectile_direction_clock.tick()

        # Create sprite list as pygame.sprite.Group() objects
        self.projectile_list = pygame.sprite.Group()
        self.all_sprites_list = pygame.sprite.Group()

        # Generate the list of projectiles
        for i in range(NUMBER_OF_PROJECTILES):
            color = random.choice(PROJECTILE_COLORS)
            projectile = Projectile(color, self.width, self.height)

            projectile.set_pos()

            self.projectile_list.add(projectile)
            self.all_sprites_list.add(projectile)

        self.player_x_movement = self.player_speed
        self.player_y_movement = 0
        # Create the player
        self.player = Player(self.width, self.height, self.player_speed)
        self.player.set_pos()
        self.all_sprites_list.add(self.player)

    def get_keys(self):
        return pygame.key.get_pressed()

    def process_events(self):
        """ Process events (eg. keystrokes). """
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                if self.game_over:
                    self.__init__()

        keys = self.get_keys()#pygame.key.get_pressed()

        self.player_x_movement, player_y_movement = self.player.get_movement()

        if keys[K_UP]:
            self.player_y_movement = -self.player_speed
            if keys[K_LEFT]:
                self.player_x_movement = -(self.player_speed)/math.sqrt(2)
                self.player_y_movement = -(self.player_speed)/math.sqrt(2)
            elif keys[K_RIGHT]:
                self.player_x_movement = (self.player_speed)/math.sqrt(2)
                self.player_y_movement = -(self.player_speed)/math.sqrt(2)
            else:
                self.player_x_movement = 0
        if keys[K_DOWN]:
            self.player_y_movement = self.player_speed
            if keys[K_LEFT]:
                self.player_x_movement = -(self.player_speed)/math.sqrt(2)
                self.player_y_movement = (self.player_speed)/math.sqrt(2)
            elif keys[K_RIGHT]:
                self.player_x_movement = (self.player_speed)/math.sqrt(2)
                self.player_y_movement = (self.player_speed)/math.sqrt(2)
            else:
                self.player_x_movement = 0
        if keys[K_LEFT]:
            self.player_x_movement = -self.player_speed
            if keys[K_UP]:
                self.player_x_movement = -(self.player_speed)/math.sqrt(2)
                self.player_y_movement = -(self.player_speed)/math.sqrt(2)
            elif keys[K_DOWN]:
                self.player_x_movement = -(self.player_speed)/math.sqrt(2)
                self.player_y_movement = (self.player_speed)/math.sqrt(2)
            else:
                self.player_y_movement = 0
        if keys[K_RIGHT]:
            self.player_x_movement = self.player_speed
            if keys[K_UP]:
                self.player_x_movement = (self.player_speed)/math.sqrt(2)
                self.player_y_movement = -(self.player_speed)/math.sqrt(2)
            elif keys[K_DOWN]:
                self.player_x_movement = (self.player_speed)/math.sqrt(2)
                self.player_y_movement = (self.player_speed)/math.sqrt(2)
            else:
                self.player_y_movement = 0

        self.player.update_movement(self.player_x_movement, self.player_y_movement)

    def logic(self):
        """ Checks if the player has collided with the objects,
        if he/she has, set game_over to True. Also updates the player's move_x
        and move_y attributes through player.update_movement(). """

        self.time_since_last_tick += self.score_clock.tick()
        if self.time_since_last_tick >= self.time_to_score_update:
            self.score += 1
            self.time_since_last_tick = 0

        self.time_scince_last_direction_switch += self.projectile_direction_clock.tick()
        if self.time_scince_last_direction_switch >= self.time_between_direction_switches:
            self.time_scince_last_direction_switch = 0
            #for projectile in self.projectile_list:
                #projectile.direction_switch()

        self.all_sprites_list.update()

        # Get a list of collisions between the projectiles and the player
        projectiles_hit = pygame.sprite.spritecollide(self.player, self.projectile_list, False)

        # If a projectile has been hit, the game is over
        if len(projectiles_hit) > 0 or self.player.hit_edge():
            if self.score > self.high_score:
                # if the current score is greater than the high score,
                # the high score is now set to the current score
                self.high_score = self.score
            self.game_over = True

    def render(self, screen):
        """ Display everything. """
        screen.fill(BROWN)

        if self.game_over:
            game_over_font = pygame.font.SysFont('freesansbold.ttf', 150)
            game_over_surf = game_over_font.render("Game Over", True, WHITE)
            game_over_rect = game_over_surf.get_rect()
            game_over_rect.midtop = (WINDOW_WIDTH / 2, WINDOW_HEIGHT / 8)
            screen.blit(game_over_surf, game_over_rect)
            high_score_font = pygame.font.Font('freesansbold.ttf', 60)
            score_surf = high_score_font.render('Final Score: %s' % (self.score), True, WHITE)
            score_rect = score_surf.get_rect()
            score_rect.midtop = (WINDOW_WIDTH / 2, ((WINDOW_HEIGHT / 8) * 3))
            screen.blit(score_surf, score_rect)
            high_score_surf = high_score_font.render('High Score: %s' % (self.high_score), True, WHITE)
            high_score_rect = high_score_surf.get_rect()
            high_score_rect.midtop = (WINDOW_WIDTH / 2, ((WINDOW_HEIGHT / 8) * 4))
            screen.blit(high_score_surf, high_score_rect)
            while self.game_over:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        self.game_over = False
                pygame.display.flip()
            self.__init__()

        if not self.game_over:
            self.all_sprites_list.draw(screen)
            score_surf = self.score_font.render('Score: %s' % (self.score), True, WHITE)
            score_rect = score_surf.get_rect()
            score_rect.topleft = (WINDOW_WIDTH - (score_rect.width + 50), 20)
            screen.blit(score_surf, score_rect)

        pygame.display.flip()



class Projectile(pygame.sprite.Sprite):
    """ Object on the screen that the player has to avoid. """

    def __init__(self, color, width, height):
        """ Initalise the block and create the image of it. """

        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        if color == BLUE:
            self.max_speed = 2
        else:
            self.max_speed = 6    
        self.direction_switch()
        

    def direction_switch(self):
        """ Switch the direction of the projectile. """

        self.move_x = random.randint(-self.max_speed, self.max_speed)
        self.move_y = random.randint(-self.max_speed, self.max_speed)
        while self.move_x == 0 or self.move_y == 0:
            self.move_x = random.randint(-self.max_speed, self.max_speed)
            self.move_y = random.randint(-self.max_speed, self.max_speed)

    def set_pos(self):
        """ Called to set the position when the game starts or resets. """

        self.rect.centerx = random.randint(self.rect.width / 2, WINDOW_WIDTH - self.rect.width / 2)
        self.rect.centery = random.randint(self.rect.width / 2, WINDOW_HEIGHT - self.rect.height / 2)

    def hit_edge(self):
        """ Makes the projectile bounce off the side of the screen. """

        

        # !!!!!!
        # WIP
        # !!!!!
        if self.rect.x <= 0 or self.rect.x + self.rect.width >= WINDOW_WIDTH:
            self.move_x = -self.move_x
        elif self.rect.y <= 0 or self.rect.y + self.rect.height >= WINDOW_HEIGHT:
            self.move_y = -self.move_y

    def get_dist(self, player):
        dist_x = (self.rect.centerx - player.rect.centerx)
        dist_y = (self.rect.centery - player.rect.centery)
        return [math.sqrt(dist_x**2 + dist_y**2), dist_x, dist_y]

    def update(self):
        """ Update the projectile position based
        on the move_x and move_y attributes. If the projectile hits the edge, it bounces off. """

        self.rect.x += self.move_x
        self.rect.y += self.move_y

        self.hit_edge()

class Player(pygame.sprite.Sprite):
    """ Object on the screen that the player has to avoid. """

    def __init__(self, width, height, speed):
        """ Initialise the player square, create an image of it and position it on the screen. """

        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([width, height])
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect()

        # pixels per frame that the player will travel
        self.speed = speed
        self.move_x = self.speed
        self.move_y = 0
        self.rect.midtop = (random.randint(200, WINDOW_WIDTH - 201), random.randint(200, WINDOW_WIDTH - 201))

    def hit_edge(self):
        """ Returns true of the player has hit the edge. """

        if (self.rect.x <= 0 or self.rect.x + self.rect.width >= WINDOW_WIDTH or self.rect.y <= 0 or self.rect.y + self.rect.height >= WINDOW_HEIGHT):
            return True

        return False

    def set_pos(self):
        """ Called to set the position when the game starts or resets. """

        self.rect.centerx = random.randint(self.rect.width / 2, WINDOW_WIDTH - self.rect.width / 2)
        self.rect.centery = random.randint(self.rect.width / 2, WINDOW_HEIGHT - self.rect.height / 2)

    def get_movement(self):
        return self.move_x, self.move_y

    def update_movement(self, x_speed, y_speed):
        self.move_x = x_speed
        self.move_y = y_speed

        
    def update(self):
        """ Update the projectile position based
        on the move_x and move_y attributes. """
        self.rect.x += self.move_x
        self.rect.y += self.move_y

if __name__ == "__main__":
    main()