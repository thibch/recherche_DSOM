# -*- coding: utf-8 -*-
### 
###   Interface graphique de som
###   basée sur cv2
### 
import classifieurs.network as net
import cv2
import numpy as np
import math


class GuiSom():        
    ''' 
    Fenêtre de suivi d'une SOM
    :Parameters:
    `som` : Classe de réseau disponible dans classifieur.network
    'width' : taille de la fenêtre de rendu du réseau
    'mainFrame' : fenêtre de capture opencv
    '''
    
    @staticmethod
    def raz_neuron(width):
        frame = np.full((width, width, 3), 200, np.uint8)    
        return frame


    def display_neuron(self, neuron):
        vector = self._network.codebook[neuron[0], neuron[1]]
        largeur = self._widthNeuron
        
        positionx = neuron[0] * largeur
        positiony = neuron[1] * largeur
    
        ## Mise à blanc du neurone
        neurframe = self.raz_neuron(self._widthNeuron)
    
        if neuron == self._winner:
            color = (0, 0, 255)
        else:
            color = (0, 0, 0)

    
        cv2.rectangle(neurframe, (0,0), (largeur-1, largeur-1), color)
    
        ## Affichage numéro neuron        
        cv2.putText(neurframe, "neur {}".format(neuron[0]*self._shape[0]+neuron[1]), (3, largeur -3), cv2.FONT_HERSHEY_SIMPLEX, .3,
                    color, 1)
        ## Affichage nb vainqueur neuron       
        cv2.putText(neurframe, "win {}".format(self._win[neuron]), (largeur -30, largeur -3), cv2.FONT_HERSHEY_SIMPLEX, .2,
                    color, 1)
 


        ## Affichage des points de saillance
        i = 0
        while i < len(vector):
            #print("vector[",i,"] = (", vector[i], ",",vector[i+1],")")
            px = int(round(largeur * vector[i]))
            py = int(round(largeur * vector[i+1]))
            cv2.circle(neurframe, (px, py), 1, 100, -1)
            ## if (i>24 and i<36) or i < 14:# or i >20:
            ##     cv2.putText(neurframe, "{}".format(int(i/2)), (px, py), cv2.FONT_HERSHEY_SIMPLEX, .5,
            ##         color, 1)
            i = i+2
        ## remplacer le neurone dans frame
        self._frame[positionx:(positionx + largeur),positiony:(positiony + largeur)] = neurframe
    
    
    def display_map (self):
        # print(" [Som Gui] display_map")
        for i in range(self._shape[0]):
            for j in range (self._shape[1]):
                self.display_neuron((i,j))

                
    def init_minidisp(self):
        largeur = self._width
        # print(" [Som Gui] minidisp, largeur = ",  self._width)
        frame = np.full((largeur, largeur, 3), 250, np.uint8)
        return frame

    
    def track_elast(self, val):
        self._elasticity=float(val)
        print("Nouvelle élasticité : ", self._elasticity, "                                          ")

    def track_lrate(self, val):
        self._lrate=float(val/100)
        print("Nouveau taux d'apprentissage : ", self._lrate, "                                          ")

    # ------------------GuiSom::__init__
    def __init__(self, shape, elasticity, lrate, initMethod, classe="DSOM", width=500, name="som"):
        self._width = width
        self._name = name
        self._elasticity = elasticity
        self._lrate = lrate;
 
        self._frame = self.init_minidisp()
        self._shape = shape
        self._widthNeuron = int(math.floor(self._width / shape[0]))
        if  shape[0] *  self._widthNeuron > self._width :
            print("Problème calcul taille du neurone")
            exit()

        self._winner = None
        self._distance   = None
        self.init_som(initMethod, classe)
        # creer fenetre som
        self.create_som_view()
        
    def init_som(self, initMethod, classe):
        module = __import__("classifieurs.network", globals(), locals(), classe)        
        classeNetwork = getattr(module, classe)
        print("\t Réseau : {0} : {1}".format(classe, self._shape)) 
        self._network = classeNetwork(self._shape, init_method=initMethod, elasticity=self._elasticity)
        self._win = np.zeros(self._shape[0:2]) # mémorise le nombre de fois ou chaque neurone est vainqueur
        self._last_win = np.zeros(self._shape[0:2]) # mémorise le nombre de fois ou chaque neurone est vainqueur depuis dernière initialisation
        # cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        # cv2.createTrackbar("elasticity", self._name, int(self._elasticity), 20, self.track_elast)
        # cv2.createTrackbar("learning rate (x100)", self._name , int(100*self._lrate), 200, self.track_lrate)
        #cv2.startWindowThread()
        #cv2.imshow(self._name, self._frame)
        #cv2.waitKey()

        
    def get_and_raz_max_last_win():
        last_winner =  np.unravel_index(np.argmax(self._last_win), self._last_win.shape)
        self._last_win[last_winner] = 0
        return last_winner

    def raz_last_win():
        self._last_win = np.zeros(self._shape[0:2])
        
    def get_and_raz_last_win(self):
        last_winner_1 =  np.unravel_index(np.argmax(self._last_win), self._last_win.shape)
        self._last_win[last_winner_1] = 0
        last_winner_2 =  np.unravel_index(np.argmax(self._last_win), self._last_win.shape)        
        self._last_win = np.zeros(self._shape[0:2])
        return last_winner_1, last_winner_2
                                  
    def set_win(self):
        return np.unravel_index(np.argmax(self._win), self._win.shape)

    

    def learn(self, data):
        sigma = 0
        # print(" [Som Gui] learn")
        self._winner, self._distance = self._network.learn_data(data, self._lrate, sigma, self._elasticity)
        # print(" [Som Gui] learn... winner = ",  self._winner)        
        self._win[self._winner] += 1
        self._last_win[self._winner] += 1
        # print(" [Som Gui] display_map()")
        self.display_map()
        # print(" [Som Gui] destroyWindow()", self._name)
        #cv2.destroyWindow(self._name)
        # print(" [Som Gui] cv2.imshow() frame = ", len(self._frame))
        #cv2.imshow(self._name, self._frame)

        #cv2.waitKey(1)
        # print(" [Som Gui] cv2.imshow()... exit")
        
    def print_win(self):
        print("Répartition des vainqueurs")
        print(self._win)
    
    def print_last_win(self):
        print("Répartition des derniers vainqueurs")
        print(self._last_win)
    

    # ------------------GuiSom::network
    @property
    def network(self):
        return self._network

    @property
    def distance(self):
        if self._distance:
            return self._distance
        else:
             return -1

    @property
    def winner(self):
        if self._winner:
            return self._winner 
        else:
             return -1

    @property
    def win(self):
        return self._win
    @property
    def frame(self):
        return self._frame

    def create_som_view(self):
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("elasticity", self._name, int(self._elasticity), 20, self.track_elast)
        cv2.createTrackbar("learning rate (x100)", self._name , int(100*self._lrate), 200, self.track_lrate)

    def show_som_view(self):        
        cv2.imshow(self._name, self._frame)

    def destroy_som_view(self):
        cv2.destroyWindow(self._name)


