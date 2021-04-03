# -*- coding: utf-8 -*-
### 
###   Interface graphique de err_som
###   basée sur cv2
### 
import classifieurs.network as net
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('agg')


class ErrGuiSom():
    ''' 
    Fenêtre de suivi d'erreur d'une SOM
    :Parameters:
    `som` : Classe de réseau disponible dans classifieur.network
    'width' : taille de la fenêtre de rendu du réseau
    'mainFrame' : fenêtre de capture opencv
    '''

    def init_minidisp(self):
        largeur = self._width
        # print(" [Som Gui] minidisp, largeur = ",  self._width)
        frame = np.full((largeur, largeur, 3), 250, np.uint8)
        return frame

    def plot_err(self, current_error=0):
        fig = plt.figure()

        self._global_errors.append(self._X * self._global_errors[-1] + (1-self._X) * current_error)

        if self._errview:
            #affiche que les 50 derniers points
            start_index = len(self._global_errors) - 50 if len(self._global_errors) > 50 else 0
            plt.plot(self._global_errors[start_index:len(self._global_errors)], 'g')
            plt.axis([0, 50, 0, 1])
            plt.gca().set_xlim(0, 50)
        else :
            plt.plot(self._global_errors, 'g')
            plt.axis([0, 500, 0, 1])
            plt.gca().set_xlim(0, len(self._global_errors))

        plt.grid()
        plt.ylabel("Taux d'erreur")
        plt.xlabel("Nombre d'itérations")

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self._frame = img
        plt.close(fig)

    def changeView(self):
        self._errview = not self._errview


    # ------------------ErrGuiSom::__init__
    def __init__(self, width=500, name="err_som"):
        self._width = width
        self._name = name

        self._errview = False
        self._global_errors = [0.0]
        self._X = 0.09

        self._frame = self.init_minidisp()
        # creer fenetre som
        self.create_err_som_view()

        self.plot_err()

    

    # ------------------ErrGuiSom::frame

    def create_err_som_view(self):
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)

    def show_err_som_view(self):
        cv2.imshow(self._name, self._frame)

    def destroy_err_som_view(self):
        cv2.destroyWindow(self._name)


