{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "bca6dd2c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "cba3a653",
   "metadata": {},
   "outputs": [],
   "source": [
    "#loading the dataset\n",
    "#this is a pixel data for a clothing article\n",
    "fashion_mnist=keras.datasets.fashion_mnist\n",
    "(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "74f951e8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_images.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0aec7ac0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000,)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_labels.shape"
   ]
  },
  {
   "cell_type": "raw",
   "id": "ba218075",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fab063de",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 28, 28)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_images.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a74ff3f8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000,)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_labels.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2a643bc8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "194"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_images[0,23,23]#look at one pixel rep from 0-255 rgb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ae9cb029",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([9, 0, 0, 3, 0, 2, 7, 2, 5, 5], dtype=uint8)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_labels[:10]#look at the first 10 training labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "336b2e22",
   "metadata": {},
   "outputs": [],
   "source": [
    "class_names=['T-shirt/top','Trouser','pullover','Dress','Coat'\n",
    "            ,'Sandal','Sneaker','Bag','Ankle boot']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "8848a848",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plot an image from the dataset\n",
    "plt.figure()\n",
    "plt.imshow(train_images[1])\n",
    "plt.colorbar()\n",
    "plt.grid(False)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f28a9803",
   "metadata": {},
   "outputs": [],
   "source": [
    "#data preprocessing\n",
    "train_images= train_images / 255.0\n",
    "test_images=test_images / 255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2289c571",
   "metadata": {},
   "outputs": [],
   "source": [
    "#from tensorflow.keras import Sequential\n",
    "#from tensorflow.keras.layers import Flatten,Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "5f01093a",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=keras.Sequential([\n",
    "    keras.layers.Conv1D(filters=32,activation='relu',kernel_size=2, input_shape=(28,28)),\n",
    "    keras.layers.Flatten(),\n",
    "    keras.layers.Dense(128, activation='relu'),\n",
    "    keras.layers.Dense(10, activation='softmax')\n",
    "])\n",
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7e584c47",
   "metadata": {},
   "outputs": [],
   "source": [
    "#lets see if it will work now"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "e7c336ae",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/3\n",
      "1875/1875 [==============================] - 249s 133ms/step - loss: 0.4553 - accuracy: 0.8380\n",
      "Epoch 2/3\n",
      "1875/1875 [==============================] - 72s 39ms/step - loss: 0.3317 - accuracy: 0.8792\n",
      "Epoch 3/3\n",
      "1875/1875 [==============================] - 72s 38ms/step - loss: 0.2924 - accuracy: 0.8915\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.keras.callbacks.History at 0x1b7e95a67c0>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(train_images,train_labels, epochs=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "f145038b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 4s 13ms/step - loss: 0.3304 - accuracy: 0.8819\n"
     ]
    }
   ],
   "source": [
    " test_loss, test_accuracy=model.evaluate(test_images, test_labels,verbose=1)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "cf81446b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test_accuracy 0.8819000124931335\n",
      "test_loss 0.33042269945144653\n"
     ]
    }
   ],
   "source": [
    "print('test_accuracy',test_accuracy)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "92845ac1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test_loss 0.33042269945144653\n"
     ]
    }
   ],
   "source": [
    "print('test_loss',test_loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "cb2a8655",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions=model.predict(test_images)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "12f5349b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1.8033842e-07 7.1321317e-08 6.8577400e-08 5.7895079e-08 8.0250874e-07\n",
      " 2.0954187e-03 5.0870841e-07 1.5673442e-03 1.0825806e-06 9.9633443e-01]\n"
     ]
    }
   ],
   "source": [
    "print(predictions[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "0ead14a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Trouser\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "print(class_names[np.argmax(predictions[5])])\n",
    "plt.figure()\n",
    "plt.imshow(test_images[5])\n",
    "plt.colorbar()\n",
    "plt.grid(False)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c7f3fa00",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
