import numpy as np

from bomberman_rl import Actions

print = lambda *args, **kwargs: None

class Agent:
    """ rule-based agent to solve the bomberman-environment by Lulius Lückefahr and Tom Wierbrügge """
    def __init__(self):
        self.setup()
        # hard-coding start behavior to learn the environment
        # startRulesBL = np.array([1, 1, 5, 3, 3, 3, 0, 0, 4, 5, 2, 2, 1, 1])
        # startRulesBR = np.array([3, 3, 5, 1, 1, 1, 0, 0, 4, 5, 2, 2, 3, 3])
        # startRulesTL= np.array([1, 1, 5, 3, 3, 3, 2, 2, 4, 5, 0, 0, 1, 1])
        # startRulesTR= np.array([3, 3, 5, 1, 1, 1, 2, 2, 4, 5, 0, 0, 3, 3])
        # self.startRules = np.zeros((4, len(startRulesBL)))
        # self.startRules[0] = startRulesBL
        # self.startRules[1] = startRulesBR
        # self.startRules[2] = startRulesTL
        # self.startRules[3] = startRulesTR
        # self.startPos = 0
        # self.lastCrates = None
        # self.potentialCoins = np.zeros((17, 17))
        self.lastExp = None

    def setup(self):
        self.rng = np.random.default_rng()

    def act(self, state: dict, **kwargs) -> int:
        action = 0
        # get current game-length:
        cStep = state.get("step")

        # save starting position (important for hard-coded behavior):
        # if cStep < 1:
        #     pos = state.get("self_pos")
        #     right = np.max([np.argmax(pos[:, n]) for n in range(17)]) > 8   # boolean x value
        #     up = np.max([np.argmax(pos[n, :]) for n in range(17)]) < 8      # boolean y value
        #     self.startPos = 0 if (not right) and (not up) else 1 if right and (not up) else 2 if not right else 3

        # keeping track of coin positions:
        # if(cStep == 0):
        #     self.lastCrates = state.get("crates")
        # currentCrates = state.get("crates")
        # self.potentialCoins += self.lastCrates - currentCrates
        # self.potentialCoins *= (np.ones((17, 17)) - state.get("opponents_pos") - state.get("self_pos"))     # contains 0 where players are, so no coins can be there anymore
        # self.lastCrates = currentCrates.copy()

        # decide for next step:
        # if cStep < len(self.startRules[0]):
        #     action = self.startRules[self.startPos, cStep]
        # else:
        pos = state.get("self_pos")
        pX = np.max([np.argmax(pos[:, n]) for n in range(17)])   # numerical (from west to east) x value
        pY = np.max([np.argmax(pos[n, :]) for n in range(17)])      # numerical (from north to south) y value   

        # estimating positionValues manually
        positionValues = np.zeros((17, 17))
        cratePos = state.get("crates")
        positionValues += 4*cratePos
        oppPos = state.get("opponents_pos")
        positionValues += 3*oppPos    # let's get some action
        #positionValues += 7*self.potentialCoins
        coinPos = state.get("coins")
        positionValues += 12*coinPos

        # get wall positions
        wallPos = state.get("walls")

        # blur positionValues
        for i in range(5):
            toConsider = np.zeros((17, 17, 7))
            for iXless, iYless in np.ndindex((15, 15)):
                iX, iY = iXless+1, iYless+1
                if wallPos[iX, iY] < 1:
                    toConsider[iX, iY] = np.array([positionValues[iX-1, iY], positionValues[iX+1, iY], positionValues[iX, iY-1], positionValues[iX, iY+1], positionValues[iX, iY], positionValues[iX, iY], positionValues[iX, iY]])
            for iX, iY in np.ndindex((17, 17)):
                positionValues[iX, iY] = toConsider[iX, iY, :].mean()

        # discouraging positions around bombs (here being surrounded by crates or players is negative)
        bombPos = state.get("bombs")
        positionValues -= 420*bombPos
        for iX, iY in np.ndindex((17, 17)):
            if bombPos[iX, iY] > 0:
                for jX in np.array(range(7))+iX-3:
                    if(jX < 1 or jX > 16):
                        continue
                    positionValues[jX, iY] = (-10+3*np.abs(jX-iX)) * np.abs(positionValues[jX, iY]) - 42*(10-3*np.abs(jX-iX))    # around bombs, values are negative
                for jY in np.array(range(7))+iY-3:
                    if(jY < 1 or jY > 16):
                        continue
                    positionValues[iX, jY] = (-10+3*np.abs(jY-iY)) * np.abs(positionValues[iX, jY]) - 42*(10-3*np.abs(jX-iX))     # around bombs, values are negative

        # make explosions and walls bad places
        if cStep < 1:
            self.lastExp = state.get("explosions").copy()
        else:
            self.lastExp = np.zeros(state.get("explosions").shape)
        expPos = state.get("explosions")
        newExp = np.max(expPos-self.lastExp, 0)    # explosions are only bad if theyre fresh
        self.lastExp = expPos
        positionValues -= 1337*newExp
        positionValues -= 1337*wallPos

        # if you are in a bomb's range, flee
        if(positionValues[pX, pY] < -1):
            #positionValues += 1337*expPos   # hoping, this will be gone, when we go there
            positionValues -= 1337*cratePos
            #positionValues -= 1337*wallPos
            positionValues -= 1337*oppPos
            positionValues -= 1337*bombPos

        # estimate best move
        actionValues = np.array([positionValues[pX, pY-1], positionValues[pX+1, pY], positionValues[pX, pY+1], positionValues[pX-1, pY], positionValues[pX, pY]])
        match (np.argmax(actionValues)):
            case 0:     #UP = 0
                if(cratePos[pX, pY-1] == 0 and oppPos[pX, pY-1] == 0):
                    action = 0
                else:
                    action = 5      #BOMB = 5
            case 1:     #RIGHT = 1
                if(cratePos[pX+1, pY] == 0 and oppPos[pX+1, pY] == 0):
                    action = 1
                else:
                    action = 5
            case 2:     #DOWN = 2
                if(cratePos[pX, pY+1] == 0 and oppPos[pX, pY+1] == 0):
                    action = 2
                else:
                    action = 5
            case 3:     #LEFT = 3
                if(cratePos[pX-1, pY] == 0 and oppPos[pX-1, pY] == 0):
                    action = 3
                else:
                    action = 5
            case 4:     #WAIT = 4
                if(min(actionValues < 0)):
                    action = 4
                else:
                    action = 5      # when he is surrounded by good spaces, be is probably between boxes, so lay a bomb
            
        #print(actionValues)
        #print(action)

        return action
