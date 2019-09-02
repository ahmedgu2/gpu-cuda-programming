# Image Blurring

In this kernel we are blurring an image. To do this, imagine that we have
a square array of weight values. For each pixel in the image, imagine that we
overlay this square array of weights on top of the image such that the center
of the weight array is aligned with the current pixel. To compute a blurred
pixel value, we multiply each pair of numbers that line up. In other words, we
multiply each weight with the pixel underneath it. Finally, we add up all of the
multiplied numbers and assign that value to our output for the current pixel.
We repeat this process for all the pixels in the image.

****************************************************************************

For a color image that has multiple channels, we seperate
the different color channels so that each color is stored contiguously
instead of being interleaved. This will simplify your code.

That is instead of RGBARGBARGBARGBA... we suggest transforming to three
arrays :
1) RRRRRRRR...
2) GGGGGGGG...
3) BBBBBBBB...

The original layout is known an Array of Structures (AoS) whereas the
format we are converting to is known as a Structure of Arrays (SoA).

Example
-------

Here is an example of computing a blur, using a weighted average, for a single
pixel in a small image.

Array of weights:

  0.0  0.2  0.0
  0.2  0.2  0.2
  0.0  0.2  0.0

Image (note that we align the array of weights to the center of the box):

    1  2  5  2  0  3
       -------
    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
      |       |
    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
      |       |
    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
       -------
    9  6  5  0  3  9

         (1)                         (2)                 (3)

