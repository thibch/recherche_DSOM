# -*- coding: utf-8 -*-
###
###   Interface graphique de network_som
###   basée sur cv2
###
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import time as t


matplotlib.use('agg')


class NetworkGuiSom():

	def update(self, distanceNetwork, winner):

		if (round(t.time() * 1000) - self._time) >= self._timeDisplayNetwork * 1000 :
			fig = plt.figure()
			shape = distanceNetwork.shape
			for i in range(shape[0]):
				for j in range(shape[1]):
					if(i < shape[0] - 1):
						x = (distanceNetwork[i, j , 0], distanceNetwork[i + 1, j , 0])
						y = (distanceNetwork[i, j , 1], distanceNetwork[i + 1, j , 1])
						plt.plot(x,y,'b',zorder = 1)
					if(j < shape[1] - 1 ):
						x = (distanceNetwork[i, j , 0], distanceNetwork[i, j + 1, 0])
						y = (distanceNetwork[i, j , 1], distanceNetwork[i, j + 1, 1])
						plt.plot(x,y,'b',zorder = 1)

			plt.scatter(distanceNetwork[:, :, 0], distanceNetwork[:, :, 1], zorder = 2)
			plt.scatter(distanceNetwork[winner[0], winner[1], 0], distanceNetwork[winner[0], winner[1], 1], c ='r', zorder = 2)


			fig.canvas.draw()
			img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
			img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			self._frame = img
			plt.close(fig)
			self._time = round(t.time() * 1000)

	# ------------------NetworkGui::__init__
	def __init__(self, width=500, name="network"):
		self._width = width
		self._timeDisplayNetwork = 2
		self._time = round(t.time() * 1000)
		self._name = name
		self._frame = self.init_minidisp()
		# creer fenetre som
		self.create_view()

	# ------------------NetworkGui::frame

	def init_minidisp(self):
		largeur = self._width
		# print(" [Som Gui] minidisp, largeur = ",  self._width)
		frame = np.full((largeur, largeur, 3), 250, np.uint8)
		return frame

	def track_refreshTime(self, val):
		self._timeDisplayNetwork = val
		print("Nouveau temps de rafraîchissement : ", self._timeDisplayNetwork, "                                          ")

	def create_view(self):
		cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
		cv2.createTrackbar("Refresh time (sec)", self._name, self._timeDisplayNetwork, 10, self.track_refreshTime)

	def show_view(self):
		cv2.imshow(self._name, self._frame)

	def destroy_view(self):
		cv2.destroyWindow(self._name)

