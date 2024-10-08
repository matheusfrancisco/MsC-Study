#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  // Read the image file
  Mat image = imread("../image.jpg");
  // Check for failure
  if (image.empty()) {
    cout << "Could not open or find the image" << endl;
    cin.get(); // wait for any key press
    return -1;
  }

  String windowName = "E.G"; // Name of the window

  namedWindow(windowName); // Create a window

  imshow(windowName, image); // Show our image inside the created window.

  waitKey(0); // Wait for any keystroke in the window

  destroyWindow(windowName); // destroy the created window

  return 0;
}
