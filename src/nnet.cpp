/* 
 * This file is part of ftlml-cv-demo.
 *
 * ftlml-cv-demo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ftlml-cv-demo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ftlml-cv-demo.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributor(s):
 * 
 * Thomas Quintana <quintana.thomas@gmail.com>
 *
 *
 * nnet.c -- A computer vision demo using convolutional neural networks.
 *
 */

#include <caffe/caffe.hpp>
#include <cuda_runtime.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdlib.h>

using namespace caffe;

unsigned int load_labels(char* path, char*** labels) {
  unsigned int n_labels = 0;
  FILE* handle = fopen(path, "r");
  // Count the number of lines (labels).
  char input;
  do {
    input = getc(handle);
    if(input == '\n' || input == EOF) {
      n_labels++;
    }
  } while(input != EOF);
  rewind(handle);
  // Load the labels into memory.
  char** list = (char**)malloc(sizeof(char*) * n_labels);
  for(unsigned int index = 0; index < n_labels; index++) {
    fpos_t position;
    fgetpos(handle, &position);
    unsigned int length = 0;
    do {
      input = getc(handle);
      if(input != '\n' && input != EOF) {
        length++;
      }
    } while(input != '\n' && input != EOF);
    fsetpos(handle, &position);
    list[index] = (char*)malloc(sizeof(char) * length + 1);
    for(unsigned int offset = 0; offset < length; offset++) {
      list[index][offset] = getc(handle);
    }
    list[index][length] = '\0';
    getc(handle); // Ignore the newline character.
  }
  *labels = list;
  return n_labels;
}

void destroy_labels(char** labels, unsigned int n_labels) {
  for(unsigned int index = 0; index < n_labels; index++) {
    free(labels[index]);
  }
  free(labels);
}

char* get_prediction(char** labels, unsigned int n_labels, float* output) {
  unsigned int max_index = 0;
  float max = 0.0f;
  for(unsigned int index = 0; index < n_labels; index++) {
    if(max < output[index]) {
      max_index = index;
      max = output[index];
    }
  }
  return labels[max_index];
}

void add_caption(IplImage* cropped_frame, char* prediction) {
  CvFont font;
  cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.5, 1.5, 0, 2, 8);
  cvPutText(cropped_frame, prediction, cvPoint(10, 30), &font, CV_RGB(255, 255, 255));
}

int main(int argc, char** argv) {
  if(argc != 4) {
    printf("Usage: nnet path/to/prototxt path/to/caffemodel path/to/labels\n");
    exit(1);
  }
  // Initialize Caffe.
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);
  // Initilize the neural network.
  Net<float> cnn(argv[1], TEST);
  cnn.CopyTrainedLayersFrom(argv[2]);
  float mean[3] = {104, 117, 123};
  // Load the label set.
  char** labels;
  unsigned int n_labels = load_labels(argv[3], &labels);
  // Allocate resources for interacting with Caffe.
  Blob<float> blob(1, 3, 224, 224);
  vector<Blob<float>*> input;
  float output[n_labels];
  // Allocate resources for frame handling.
  IplImage* cnn_input = cvCreateImage(cvSize(224, 224), 8, 3);
  IplImage* cropped_frame = cvCreateImage(cvSize(480, 480), 8, 3);
  // Open the camera for capturing frames.
  CvCapture* capture = cvCaptureFromCAM(0);
  // Create a display window.
  cvNamedWindow("Ft. Lauderdale Machine Learning Meetup");
  while(1) {
    cvGrabFrame(capture);
    IplImage* frame = cvRetrieveFrame(capture);
    // Crop the center of the frame to 480x480.
    int x = (frame->width - 480) / 2;
    int y = (frame->height - 480) / 2;
    CvRect crop_rect = cvRect(x, y, 480, 480);
    cvSetImageROI(frame, crop_rect);
    cvCopy(frame, cropped_frame, NULL);
    cvResetImageROI(frame);
    // Prepare the frame as an input to the neural network.
    cvResize(cropped_frame, cnn_input);
    // Subtract the mean from from the images and copy into a caffe blob.
    for(unsigned char channel = 0; channel < 3; channel++) {
      for(y = 0; y < 224; y++) {
        for(x = 0; x < 224; x++) {
          unsigned char pixel = cnn_input->imageData[y * cnn_input->widthStep + x * cnn_input->nChannels + channel];
          blob.mutable_cpu_data()[blob.offset(0, channel, y, x)] = (float)(pixel - mean[channel]);
        }
      }
    }
    input.push_back(&blob);
    // Feed the blob through the neural network.
    float loss;
    const vector<Blob<float>*>& result = cnn.Forward(input, &loss);
    input.clear();
    // Copy the resuls out of the blob.
    for(unsigned int index = 0; index < n_labels; index++) {
      output[index] = result[0]->cpu_data()[index];
    }
    // Display the results.
    char* prediction = get_prediction(labels, n_labels, output);
    add_caption(cropped_frame, prediction);
    cvShowImage("Ft. Lauderdale Machine Learning Meetup", cropped_frame);
    // If the user hits Esc end the prediction loop.
    if(cvWaitKey(50) == 27) {
      break;
    }
  }
  // Cleanup
  cvReleaseCapture(&capture);
  cvReleaseImage(&cropped_frame);
  cvReleaseImage(&cnn_input);
  destroy_labels(labels, n_labels);
  cvDestroyWindow("Ft. Lauderdale Machine Learning Meetup");
  exit(0);
}