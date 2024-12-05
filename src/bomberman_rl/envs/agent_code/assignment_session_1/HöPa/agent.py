from bomberman_rl import Actions
import numpy as np
from collections import deque
from random import shuffle

def find_rescue_route(bomb_exploison_radius,crates,walls,explosions,opponents_pos,i,j,iteration,memo,limit):
    if(((i,j,iteration) in memo)):
        return 
    memo[(i,j,iteration)]=1
    if(iteration>0 and (crates[i,j]==1 or walls[i,j]==1 or explosions[i,j]-iteration>0 or bomb_exploison_radius[i,j]+iteration>=5 or ((i,j) in opponents_pos))):
        memo[(i,j,iteration)]=0
           
    elif(bomb_exploison_radius[i,j]>=1):
        memo[(i,j,iteration)]=0
        if(iteration<limit):

            find_rescue_route(bomb_exploison_radius,crates,walls,explosions,opponents_pos,i-1,j,iteration+1,memo,limit)
            memo[(i,j,iteration)]=max(memo[(i,j,iteration)],memo[(i-1,j,iteration+1)])
            if iteration==0 and memo[(i,j,iteration)]==1:
                return 3

            find_rescue_route(bomb_exploison_radius,crates,walls,explosions,opponents_pos,i+1,j,iteration+1,memo,limit)
            memo[(i,j,iteration)]=max(memo[(i,j,iteration)],memo[(i+1,j,iteration+1)])
            if iteration==0 and memo[(i,j,iteration)]==1:
                return 1

            find_rescue_route(bomb_exploison_radius,crates,walls,explosions,opponents_pos,i,j-1,iteration+1,memo,limit)
            memo[(i,j,iteration)]=max(memo[(i,j,iteration)],memo[(i,j-1,iteration+1)])
            if iteration==0 and memo[(i,j,iteration)]==1:
                return 0

            find_rescue_route(bomb_exploison_radius,crates,walls,explosions,opponents_pos,i,j+1,iteration+1,memo,limit)
            memo[(i,j,iteration)]=max(memo[(i,j,iteration)],memo[(i,j+1,iteration+1)])
            if iteration==0 and memo[(i,j,iteration)]==1:
                return 2

            find_rescue_route(bomb_exploison_radius,crates,walls,explosions,opponents_pos,i,j,iteration+1,memo,limit)
            memo[(i,j,iteration)]=max(memo[(i,j,iteration)],memo[(i,j,iteration+1)])
            if iteration==0 and memo[(i,j,iteration)]==1:
                return 4


      
class Agent:




    
    def __init__(self):
        self.setup()


    def setup(self):
            pass

   


    def reset_self(self):
        pass

   



    def act(self, game_state: dict, **kwargs) -> int:
        """
        Called each game step to determine the agent's next action.

        You can find out about the state of the game environment via game_state,
        which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
        what it contains.
        """
   
        
        # Gather information about the game state
        walls = game_state["walls"]
        position = game_state["self_info"]["position"]
        bombs_left = game_state["self_info"]["bombs_left"]
        score= game_state["self_info"]["score"]
        #berechne eigene Position
        (x,y)=(0,0)
        for a in range(0,17):
             for b in range(0,17):
                if position[a,b]==1:
                    (x,y)=(a,b)


        bombs = game_state["bombs"]
        #Koordinaten von Bomben
        bomb_xys = [(i, j) for i in range(0,17) for j in range(0,17) if bombs[i,j]>=1]
        #print("\n Bomben:")
        #print(bombs)
        opponents=game_state["opponents_pos"]
        #Gegner Positionen
        opponents_pos=[(i, j) for i in range(0,17) for j in range(0,17) if opponents[i,j]==1]
        crates=game_state["crates"]
        explosions=game_state["explosions"]
        coins=game_state["coins"]

        for i in range(0,17):
            for j in range(0,17):
                if explosions[i,j]==12:
                    explosions[i,j]=4
                if explosions[i,j]==11:
                    explosions[i,j]=3


        #print("\n explosion:")
        #print(explosions)
        # berechne erreichbare Felder, die Felder die begehbar sind ohne zu explodieren
        parseable=np.ones((17,17))
        for i in range(0,17):
             for j in range(0,17):
                 if(crates[i,j]==1 or explosions[i,j]>1 or walls[i,j]==1):
                     parseable[i,j]=0
                 if(bombs[i,j]==4):
                    parseable[i,j]=0
                    for k in range(1,4):
                        if(i-k>0 and walls[i-k,j]==0):
                            parseable[i-k,j]=0
                        #Wand blockt Exlposion breche ab
                        elif(i-k>0 and walls[i-k,j]==1):
                            break
                    #Streifen oben
                    for k in range(1,4): 
                        if(j-k>0 and walls[i,j-k]==0):
                            parseable[i,j-k]=0
                        #Wand blockt Exlposion breche ab
                        elif(j-k>0 and walls[i,j-k]==1):
                            break

                    #Streifen rechts
                    for k in range(1,4):
                        if(i+k<16 and walls[i+k,j]==0):
                            parseable[i+k,j]=0
                        #Wand blockt Exlposion breche ab
                        elif(i+k<16 and walls[i+k,j]==1):
                            break
                    #Streifen unten
                    for k in range(1,4): 
                        if(j+k<16 and walls[i,j+k]==0):
                                parseable[i,j+k]=0
                        #Wand blockt Exlposion breche ab
                        elif(j+k<16 and walls[i,j+k]==1):
                            break

   
        


         # berechne nun Explosionsradius von gelegten aber noch nicht exlpodierten Bomben 
        bomb_exploison_radius=np.zeros((17,17))
        for (i,j) in bomb_xys:
            bomb_exploison_radius[i,j]=bombs[i,j]
            for k in range(1,4):
                if(i-k>0 and walls[i-k,j]==0):
                  bomb_exploison_radius[i-k,j]=bombs[i,j]
                #Wand blockt Exlposion breche ab
                elif(i-k>0 and walls[i-k,j]==1):
                    break
            #Streifen oben
            for k in range(1,4): 
                if(j-k>0 and walls[i,j-k]==0):
                  bomb_exploison_radius[i,j-k]=bombs[i,j]
                #Wand blockt Exlposion breche ab
                elif(j-k>0 and walls[i,j-k]==1):
                    break

            #Streifen rechts
            for k in range(1,4):
                if(i+k<16 and walls[i+k,j]==0):
                    bomb_exploison_radius[i+k,j]=bombs[i,j]
                #Wand blockt Exlposion breche ab
                elif(i+k<16 and walls[i+k,j]==1):
                    break
            #Streifen unten
            for k in range(1,4): 
                if(j+k<16 and walls[i,j+k]==0):
                     bomb_exploison_radius[i,j+k]=bombs[i,j]
                #Wand blockt Exlposion breche ab
                elif(j+k<16 and walls[i,j+k]==1):
                    break
           

        #Agent ist im explosionsradius einer Bombe, nimm kuerzesten Weg aus dem Radius zum Fliehen
        if(bomb_exploison_radius[x][y]>=1):
            memo={}
            
            direction=find_rescue_route(bomb_exploison_radius,crates,walls,explosions,opponents_pos,x,y,0,memo,10)
            if(direction==None):
                direction=4
            # print("\n")

            # print(1)
            # print(direction)
            return direction

        #in keinem Bombenradius versuche naechsten Gegner wegzusprengen
        #berechne naechsten Gegener bez. Distanz und naechstes Crate
        
        

        q = deque()
        q.append((x,y,0))
        visited=np.zeros((17,17))
        parent={}
        nearest_opponent_x=-1
        nearest_opponent_y=-1
        nearest_opponent_d=1000

        nearest_crate_x=-1
        nearest_crate_y=-1
        nearest_crate_d=1000

        nearest_coin_x=-1
        nearest_coin_y=-1
        nearest_coin_d=1000
        while(q):
            (i,j,d)=q.popleft()

            if(opponents_pos.count((i,j))>0 and nearest_opponent_d>d):
                nearest_opponent_x=i
                nearest_opponent_y=j
                nearest_opponent_d=d

            if(coins[i,j]>0 and nearest_coin_d>d):
                nearest_coin_x=i
                nearest_coin_y=j
                nearest_coin_d=d


            if((crates[i-1,j]==1 or crates[i+1,j]==1 or crates[i,j-1]==1 or crates[i,j+1]==1) and nearest_crate_d>d):
                 nearest_crate_x=i
                 nearest_crate_y=j
                 nearest_crate_d=d

            if(visited[i,j]==0):
                visited[i,j]=1
                if(parseable[i-1,j]==1 and visited[i-1,j]==0):
                    q.append((i-1,j,d+1))
                    parent[(i-1,j)]=(i,j)
                
                if(parseable[i+1,j]==1 and visited[i+1,j]==0):
                    q.append((i+1,j,d+1))
                    parent[(i+1,j)]=(i,j)
                
                if(parseable[i,j-1]==1 and visited[i,j-1]==0):
                    q.append((i,j-1,d+1))
                    parent[(i,j-1)]=(i,j)
                
                if(parseable[i,j+1]==1 and visited[i,j+1]==0):
                    q.append((i,j+1,d+1))
                    parent[(i,j+1)]=(i,j)

        if(nearest_coin_x!=-1):

            #An Crate annaehern
            current_x = nearest_coin_x
            current_y=nearest_coin_y
            (next_pos_x,next_pos_y)=(-1,-1)
            #berechne naechstes Feld
            while True:
                if parent[(current_x,current_y)] == (x,y):
                    (next_pos_x,next_pos_y)= (current_x,current_y)
                    break
                (current_x,current_y) = parent[(current_x,current_y)]
            direction=4
            if((next_pos_x,next_pos_y)==(x-1,y) ):
                direction=3
                
            elif((next_pos_x,next_pos_y)==(x+1,y) ):
                direction=1
                                
            elif((next_pos_x,next_pos_y)==(x,y-1)):
                direction=0
                                                     
            elif((next_pos_x,next_pos_y)==(x,y+1) ):
                direction=2
            # print("\n")
            # print(10)
            # print(5)
            return direction
        #Gegner ist erreichbar und Bombe vorhanden, versuche zu benutzen
        elif(nearest_opponent_d!=1000 and bombs_left==True):
            
            #pruefe nun ob Gegner im Explosionradius waere, wenn Boombe gelegt wird
            in_radius=False
            #Streifen links
            for k in range(1,4):
                if(x-k>0 and walls[x-k,y]==0 and x-k==nearest_opponent_x and y==nearest_opponent_y):
                   in_radius=True
                #Wand blockt Exlposion breche ab
                elif(x-k>0 and walls[x-k,y]==1):
                    break
            #Streifen oben
            for k in range(1,4): 
                if(y-k>0 and walls[x,y-k]==0 and x==nearest_opponent_x and y-k==nearest_opponent_y):
                   in_radius=True
                #Wand blockt Exlposion breche ab
                elif(y-k>0 and walls[x,y-k]==1):
                    break

            #Streifen rechts
            for k in range(1,4):
                if(x+k<16 and walls[x+k,y]==0 and x+k==nearest_opponent_x and y==nearest_opponent_y):
                    in_radius=True
                #Wand blockt Exlposion breche ab
                elif(x+k<16 and walls[x+k,y]==1):
                    break
            #Streifen unten
            for k in range(1,4): 
                if(y+k<16 and walls[x,y+k]==0 and x==nearest_opponent_x and y+k==nearest_opponent_y):
                    in_radius=True
                #Wand blockt Exlposion breche ab
                elif(y+k<16 and walls[x,y+k]==1):
                    break
            #im Radius und nahe genug damit es gefaehrlich wird also lege Bombe
            if(in_radius==True and nearest_opponent_d<3):
                # print("\n")
                # print(2)
                # print(5)
                return 5
            #An Gegner annaehern
            current_x = nearest_opponent_x
            current_y=nearest_opponent_y
            (next_pos_x,next_pos_y)=(-1,-1)
            #berechne naechstes Feld
            while True:
                if parent[(current_x,current_y)] == (x,y):
                    (next_pos_x,next_pos_y)= (current_x,current_y)
                    break
                (current_x,current_y) = parent[(current_x,current_y)]
            direction=4
            if((next_pos_x,next_pos_y)==(x-1,y) ):
                direction=3
                
            elif((next_pos_x,next_pos_y)==(x+1,y) ):
                direction=1
                                
            elif((next_pos_x,next_pos_y)==(x,y-1)):
                direction=0
                                                     
            elif((next_pos_x,next_pos_y)==(x,y+1) ):
                direction=2
            
            # print("\n")
            # print(3)
            # print(direction)
            return direction
       
        #keine Aktion bez. Gegner moeglich und Bombe vorhanden, also sprenge Crates Weg
        elif(bombs_left==True and nearest_crate_x!=-1):
            
            #Crate liegt neben Agenten, sprenge weg
            if(nearest_crate_d==0):
                # print("\n")
                # print(5)
                # print(5)
                return 5

            #An Crate annaehern
            current_x = nearest_crate_x
            current_y=nearest_crate_y
            (next_pos_x,next_pos_y)=(-1,-1)
            #berechne naechstes Feld
            while True:
                if parent[(current_x,current_y)] == (x,y):
                    (next_pos_x,next_pos_y)= (current_x,current_y)
                    break
                (current_x,current_y) = parent[(current_x,current_y)]
            direction=4
            if((next_pos_x,next_pos_y)==(x-1,y) ):
                direction=3
                
            elif((next_pos_x,next_pos_y)==(x+1,y) ):
                direction=1
                                
            elif((next_pos_x,next_pos_y)==(x,y-1)):
                direction=0
                                                     
            elif((next_pos_x,next_pos_y)==(x,y+1) ):
                direction=2
           
            # print("\n")
            # print(6)
            # print(direction)
            return direction
        


        else:

            # print("\n")
            # print(7)
            # print(4)
            return 4



        return 4
        