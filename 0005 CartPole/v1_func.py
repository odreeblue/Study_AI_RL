# -*- coding: utf-8 -*-
# Package import to be used for implementation(구현)
import numpy as np
import matplotlib.pyplot as plt
import gym
from JSAnimation.IPython_display import display_animation
# A function that makes animations
from matplotlib import animation


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif , with controls
    """

    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('movie_cartpole.mp4') # The part where i save the animation
    