/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <Arduino.h>
#include <TensorFlowLite.h>

#include "main_functions.h"
#include "motion_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* motion_model = nullptr;
tflite::MicroInterpreter* motion_interpreter = nullptr;
TfLiteTensor* motion_input_tensor = nullptr;
TfLiteTensor* motion_output_tensor = nullptr;
int input_length;

const int NUM_SAMPLES_PER_CHANNEL = 80;
const int NUM_CHANNELS = 3;
const int TOTAL_SAMPLES = NUM_SAMPLES_PER_CHANNEL * NUM_CHANNELS;

// Create an area of memory to use for input, output, and intermediate arrays.
// Minimum arena size, at the time of writing. After allocating tensors
// you can retrieve this value by invoking interpreter.arena_used_bytes().
const int kModelArenaSize = 100*1024;
// Extra headroom for model + alignment + future interpreter changes.
const int kExtraArenaSize = 560 + 16 + 100;
const int kTensorArenaSize = kModelArenaSize + kExtraArenaSize;
uint8_t tensor_arena[kTensorArenaSize];
float pwave[] = {
-0.10298943, -0.20157108, 0.27269047, -0.039029147, -0.21277405, 0.13626362, 0.029946933, -0.24311645, 0.05540123, 0.01766969, -0.020787034, 0.081074364, -0.21817803, 0.13519156, 0.055493187, -0.17429113, 0.06662963, -0.24176073, -0.08174706, -0.05664207, -0.267711, -0.060233407, -0.18395229, 0.18390124, 0.014912553, 0.27814344, -0.17958272, 0.12516722, -0.032570202, -0.094908506, 0.012203747, 0.00072792394, -0.098641664, -0.3404834, 1.0, -0.14334887, -0.4167804, 0.08091882, 0.22800767, -0.2526466, -0.9679784, 0.7239018, 0.12646843, -0.11565226, 0.18552037, -0.15486991, 0.25412738, -0.062063955, -0.029590018, 0.14866492, -0.046950944, 0.39673394, -0.3434065, 0.3253976, -0.60022724, 0.06339677, 0.31210408, -0.036342178, -0.63217884, 0.41075367, -0.105423935, 0.10484856, 0.23017403, -0.12573935, 0.2654194, -0.22481865, 0.38020152, -0.11052448, -0.03210346, -0.029229756, -0.110576294, 0.033547238, -0.022905761, -0.21528396, 0.33412346, 0.25225404, 0.13735217, -0.107161924, -0.3137925, 0.591233, 0.1968412, -0.18461223, 0.03237113, -0.052394424, -0.06027397, -0.0483194, -0.045872748, 0.069058925, 0.130039, 0.044414807, -0.08710753, -0.02047027, 0.00964168, -0.050019793, 0.07212054, -0.026673112, -0.024566261, -0.13166814, 0.08444434, -0.07586515, 0.013308659, 0.045130335, 0.0063777333, 0.0049250643, 0.0650618, 0.06499497, 0.004528745, -0.02724453, 0.014356015, -0.08666398, 0.07226838, -0.19566146, 0.26811454, 0.5954571, -0.8824838, -0.11368089, -0.005760246, 0.03900539, -0.64445657, 0.55141944, 1.0, -0.92442673, 0.14440618, -0.62368774, 0.1909338, 0.46166378, -0.26391, 0.42432424, 0.016255258, -0.29499894, -0.039761405, -0.17883576, 0.28489533, 0.0019892433, -0.27492544, 0.4209924, -0.3550202, 0.16866863, 0.20207325, -0.16388872, -0.15175545, -0.34494704, 0.13736618, -0.05173146, 0.26479688, 0.5167072, -0.4950229, 0.04496397, 0.07369298, 0.057236996, -0.17317498, 0.43093672, 0.0934852, -0.1375754, 0.089132704, 0.04741713, 0.24616125, -0.0771401, -0.068077385, -0.31930488, -0.09502535, 0.0030374667, -0.081569806, 0.08449756, 0.011031903, -0.013561726, -0.024886508, 0.070402816, -0.056723416, -0.045034107, 0.058829337, -0.01615657, 0.01548446, 0.024149798, 0.068574354, 0.14950033, -0.045626722, 0.18333364, 0.04340771, 0.029061226, -0.052448798, 0.0027311516, -0.13014448, 0.09264417, -0.098880835, 0.12179748, -0.01583524, 0.093984835, -0.280959, 0.26321796, -0.28194913, 0.5971714, -0.54315764, -0.7339134, -0.020587767, 1.0, 0.99355614, -1.370161, 0.132987, 0.31020397, -0.6222643, 0.35911196, -0.76859707, 0.70582885, 0.41676474, -0.8003422, -0.036780696, 0.5456929, 0.4986037, -0.037598934, 0.014272628, -0.02498151, -0.38776374, 0.07102233, -0.090301365, -0.29301152, 0.5375854, -0.39149293, -0.3232255, 0.100276954, -0.2183211, 0.58128077, -0.19898105, 0.06589507, 0.33452427, -0.70400083, 0.47127968, 0.5599736, -0.35208234, -0.30999935, -0.38004637, 0.17360051, 0.040075563, 0.18315479, -0.12482182, -0.42927462, 0.1967438, -0.30637816, 0.47997507, 0.021619868};


unsigned long time_before; //the Arduino function millis(), that we use to time out model, returns a unsigned long.
unsigned long time_after;
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  motion_model = tflite::GetModel(g_motion_model);
  if (motion_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         motion_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;


  static tflite::MicroInterpreter static_interpreter(
      motion_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  motion_interpreter = &static_interpreter;


  TfLiteStatus allocate_status = motion_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "interpreter: AllocateTensors() failed");
    return;
  }

  motion_input_tensor = motion_interpreter->input(0);

  input_length = motion_input_tensor->bytes / sizeof(float);

  motion_output_tensor = motion_interpreter->output(0);
}

// The name of this function is important for Arduino compatibility.
void loop() {

  for (int i = 0; i < TOTAL_SAMPLES; ++i) {
      motion_input_tensor->data.f[i] = pwave[i];
    }


  // Run inferencing
  time_before = millis();
  TfLiteStatus invoke_status = motion_interpreter->Invoke();
  time_after = millis();

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter,"Invoke failed!");
    while (1);
    return;
  }

  // Loop through the output tensor values from the model
  float max_val = 0.0;
  int max_index = 0;
  for (int i = 0; i < num_classes; i++) {
    if (motion_output_tensor->data.f[i] > max_val){
      max_val = motion_output_tensor->data.f[i];
      max_index = i;
    }
  }

  TF_LITE_REPORT_ERROR(error_reporter, "Time Stamp: %d", time_after);
  TF_LITE_REPORT_ERROR(error_reporter, "Class: %s", CLASSES[max_index]);
  TF_LITE_REPORT_ERROR(error_reporter, "Invoke time (mS): %d", time_after-time_before);
  TF_LITE_REPORT_ERROR(error_reporter, "Memory Consumption (bytes): %d", motion_interpreter->arena_used_bytes());
  delay(500);
}
