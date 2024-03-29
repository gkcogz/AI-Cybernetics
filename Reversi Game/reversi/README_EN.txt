python means python3

* headless_reversi_creator.py	- play the reversi game without GUI (Graphical User Interface)
* reversi_creator.py		- run GUI where you can choose various players

** headless_reversi_creator **

headless_reversi_creator accepts one or two compulsory parameters where you can specify the players. 

in a terminal:
>> python headless_reversi_creator.py player

Expects MyPlayer class in player.py. 
or
>> python headless_reversi_creator.py player player
Will run your player againts itself. You can have more players. 

>> python headless_reversi_creator.py player another_player

Expects MyPlayer class in another_player.py

You can also freely modify the source of the headless_reversi_creator if you prefer

import player

...
...
...

    if len(args) == 0:
---->   p1 = player.MyPlayer(p1_color, p2_color)
        p2 = random_player.MyPlayer(p2_color, p1_color)
        game = HeadlessReversiCreator(p1, p1_color, p2, p2_color, 8)
        game.play_game()



** reversi_creator **
You can run the interactive version of the game by

>> python reversi_creator.py player

Expects MyPlayer class in player.py.

Similarly to the non-gui version you can specify alternative players

>> python reversi_creator.py player another_player yet_another_player

Again, you can modify the source of he reversi_creator if you wish

import player

...
...
...

if __name__ == "__main__": 
--> players_dict = {'random':random_player.RandomPlayer, 'my_player':player.MyPlayer}
    game = ReversiCreator(players_dict)
    game.gui.root.mainloop()
