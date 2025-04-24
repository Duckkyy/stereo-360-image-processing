// #include <cmath>
// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <omp.h>

// using namespace std;
// using namespace cv;

// struct Vec3 {
//     float x, y, z;
//     Vec3(float a, float b, float c): x(a), y(b), z(c) {}
//     Vec3 normalize() const {
//         float len = sqrt(x*x + y*y + z*z);
//         return Vec3(x/len, y/len, z/len);
//     }
// };

// Vec3 direction_from_face_coords(int face, float u, float v) {
//     float x = 2.0f * u - 1.0f;
//     float y = 2.0f * v - 1.0f;

//     switch (face) {
//         case 0: return Vec3(1.0f, -y, -x).normalize();  // +X
//         case 1: return Vec3(-1.0f, -y, x).normalize();  // -X
//         case 2: return Vec3(x, 1.0f, y).normalize();    // +Y
//         case 3: return Vec3(x, -1.0f, -y).normalize();  // -Y
//         case 4: return Vec3(x, -y, 1.0f).normalize();   // +Z
//         case 5: return Vec3(-x, -y, -1.0f).normalize(); // -Z
//     }
//     return Vec3(0, 0, 0);
// }

// void dir_to_uv(const Vec3& dir, float& u, float& v) {
//     float theta = atan2(dir.y, dir.x);
//     float phi = acos(dir.z);
//     u = (theta + M_PI) / (2.0f * M_PI);
//     v = phi / M_PI;
// }

// void convert_face(const Mat& input, int faceSize, int face, const string& outputFilename) {
//     Mat faceImage(faceSize, faceSize, CV_8UC3);

//     #pragma omp parallel for collapse(2)
//     for (int y = 0; y < faceSize; ++y) {
//         for (int x = 0; x < faceSize; ++x) {
//             float u = (x + 0.5f) / faceSize;
//             float v = (y + 0.5f) / faceSize;

//             Vec3 dir = direction_from_face_coords(face, u, v);
//             float uf, vf;
//             dir_to_uv(dir, uf, vf);

//             int px = min(int(uf * input.cols), input.cols - 1);
//             int py = min(int(vf * input.rows), input.rows - 1);

//             Vec3b color = input.at<Vec3b>(py, px);
//             faceImage.at<Vec3b>(y, x) = color;
//         }
//     }

//     imwrite(outputFilename, faceImage);
// }

// int main(int argc, char** argv) {
//     if (argc < 2) {
//         cout << "Usage: " << argv[0] << " <input_equirectangular_image>" << endl;
//         return -1;
//     }

//     Mat input = imread(argv[1]);
//     if (input.empty()) {
//         cerr << "Failed to load image: " << argv[1] << endl;
//         return -1;
//     }

//     int faceSize = input.cols / 4;
//     const string faceNames[6] = { "XPOS", "XNEG", "YPOS", "YNEG", "ZPOS", "ZNEG" };

//     #pragma omp parallel for
//     for (int i = 0; i < 6; ++i) {
//         string filename = "CUBE_" + faceNames[i] + ".png";
//         convert_face(input, faceSize, i, filename);
//     }

//     cout << "Cube map faces saved: XPOS, XNEG, YPOS, YNEG, ZPOS, ZNEG" << endl;
//     return 0;
// }


#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

struct Vec3 {
    float x, y, z;
    Vec3(float a, float b, float c): x(a), y(b), z(c) {}
    Vec3 normalize() const {
        float len = sqrt(x*x + y*y + z*z);
        return Vec3(x/len, y/len, z/len);
    }
};

Vec3 direction_from_face_coords(int face, float u, float v) {
    float x = 2.0f * u - 1.0f;
    float y = 2.0f * v - 1.0f;

    switch (face) {
        case 0: return Vec3(1.0f, -y, -x).normalize();  // +X
        case 1: return Vec3(-1.0f, -y, x).normalize();  // -X
        case 2: return Vec3(x, 1.0f, y).normalize();    // +Y
        case 3: return Vec3(x, -1.0f, -y).normalize();  // -Y
        case 4: return Vec3(x, -y, 1.0f).normalize();   // +Z
        case 5: return Vec3(-x, -y, -1.0f).normalize(); // -Z
    }
    return Vec3(0, 0, 0);
}

void dir_to_uv(const Vec3& dir, float& u, float& v) {
    float theta = atan2(dir.y, dir.x);
    float phi = acos(dir.z);
    u = (theta + M_PI) / (2.0f * M_PI);
    v = phi / M_PI;
}

Mat convert_face(const Mat& input, int faceSize, int face) {
    Mat faceImage(faceSize, faceSize, CV_8UC3);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < faceSize; ++y) {
        for (int x = 0; x < faceSize; ++x) {
            float u = (x + 0.5f) / faceSize;
            float v = (y + 0.5f) / faceSize;

            Vec3 dir = direction_from_face_coords(face, u, v);
            float uf, vf;
            dir_to_uv(dir, uf, vf);

            int px = min(int(uf * input.cols), input.cols - 1);
            int py = min(int(vf * input.rows), input.rows - 1);

            Vec3b color = input.at<Vec3b>(py, px);
            faceImage.at<Vec3b>(y, x) = color;
        }
    }

    return faceImage;
}

Mat apply_rotation(Mat& img, const string& faceName) {
    if (faceName == "ZNEG" || faceName == "ZPOS" || faceName == "XNEG") {
        rotate(img, img, ROTATE_90_COUNTERCLOCKWISE);
    } else if (faceName == "XPOS") {
        rotate(img, img, ROTATE_90_CLOCKWISE);
    } else if (faceName == "YPOS") {
        rotate(img, img, ROTATE_180);
        // To rotate "190", you can apply affine, but 180° is sufficient unless exact needed
    }
    return img;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <equirectangular_image>" << endl;
        return -1;
    }

    Mat input = imread(argv[1]);
    if (input.empty()) {
        cerr << "Error loading image: " << argv[1] << endl;
        return -1;
    }

    int faceSize = input.cols / 4;
    map<string, int> faceMap = {
        {"XPOS", 0}, {"XNEG", 1}, {"YPOS", 2}, {"YNEG", 3}, {"ZPOS", 4}, {"ZNEG", 5}
    };

    map<string, Mat> faces;
    for (const auto& [name, index] : faceMap) {
        Mat face = convert_face(input, faceSize, index);
        faces[name] = apply_rotation(face, name);
    }

    // Combine into 2-row 3-column layout: [Y+, X+, Y-] on top, [X-, Z-, Z+] below
    Mat topRow, bottomRow;
    hconcat(vector<Mat>{faces["YPOS"], faces["XNEG"], faces["YNEG"]}, topRow);
    hconcat(vector<Mat>{faces["XPOS"], faces["ZNEG"], faces["ZPOS"]}, bottomRow);
    Mat finalImage;
    vconcat(topRow, bottomRow, finalImage);

    imwrite("CUBEMAP_LAYOUT.png", finalImage);
    cout << "Combined cubemap image saved as: CUBEMAP_LAYOUT.png" << endl;
    return 0;
}
