package com.example.audioapp;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private Button cameraButton, predictButton;
    private Bitmap currentImage;
    private static final int CAMERA_REQUEST_CODE = 1001;
    private static final int GALLERY_REQUEST_CODE = 1002;
    private static final int PERMISSION_REQUEST_CODE = 200;
    private Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraButton = findViewById(R.id.camera_Button);
        predictButton = findViewById(R.id.predict_Button);

        if (!checkPermission()) {
            requestPermission();
        }

        loadModel();

        cameraButton.setOnClickListener(v -> openCamera());

        predictButton.setOnClickListener(v -> {
            if (currentImage != null) {
                predict(currentImage);
            } else {
                openGallery();
            }
        });
    }

    private void loadModel() {
        try {
            AssetFileDescriptor fileDescriptor = this.getAssets().openFd("lenet5_model.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer tfliteModel = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            tflite = new Interpreter(tfliteModel);

            Interpreter.Options options = new Interpreter.Options();
            Interpreter tflite = new Interpreter(tfliteModel, options);
            int inputTensorIndex = 0;
            int[] inputShape = tflite.getInputTensor(inputTensorIndex).shape();
            int outputTensorIndex = 0;
            int[] outputShape = tflite.getOutputTensor(outputTensorIndex).shape();

            String inputShapeStr = Arrays.toString(inputShape);
            String outputShapeStr = Arrays.toString(outputShape);

            // Display input and output shapes using AlertDialog
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private void predict(Bitmap bitmap) {
        if (tflite == null) {
            Log.e("ModelLoading", "TensorFlow Lite model is not loaded");
            return;
        }

        // Record start time
        long startTime = System.currentTimeMillis();

        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true);

        float[][][][] input = new float[1][28][28][1];

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int pixel = resizedBitmap.getPixel(x, y);
                // Normalize pixel values to be between 0 and 1
                input[0][y][x][0] = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3.0f / 255.0f;
            }
        }

        float[][] output = new float[1][10];

        // Record start time of inference
        long inferenceStartTime = System.currentTimeMillis();

        tflite.run(input, output);

        // Record end time of inference
        long inferenceEndTime = System.currentTimeMillis();

        // Calculate latency
        long latency = inferenceEndTime - inferenceStartTime;

        // Display latency as an alert message
        String latencyMessage = "Inference Latency: " + latency + " milliseconds";
        new AlertDialog.Builder(this)
                .setTitle("Latency")
                .setMessage(latencyMessage)
                .setPositiveButton(android.R.string.ok, null)
                .show();

        int predictedDigit = getPredictedDigit(output);


        Toast.makeText(this, "Predicted Digit: " + predictedDigit, Toast.LENGTH_SHORT).show();

        // Record end time
        long endTime = System.currentTimeMillis();

        // Calculate total time (including model loading, preprocessing, and postprocessing)
        long totalTime = endTime - startTime;
        // Display total time as an alert dialog
        String totalTimeMessage = "Total Time: " + totalTime + " milliseconds";
        new AlertDialog.Builder(this)
                .setTitle("Total Time")
                .setMessage(totalTimeMessage)
                .setPositiveButton(android.R.string.ok, null)
                .show();
    }

    private int getPredictedDigit(float[][] output) {
        int predictedDigit = -1;
        float maxConfidence =0;
        for (int i = 0; i < 10; i++) {
            if (output[0][i] > maxConfidence) {
                predictedDigit = i;
                maxConfidence = output[0][i];
            }
        }

        return predictedDigit;
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, CAMERA_REQUEST_CODE);
    }

    private void openGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(galleryIntent, GALLERY_REQUEST_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == CAMERA_REQUEST_CODE && resultCode == RESULT_OK) {
            if (data != null && data.getExtras() != null) {
                Bundle extras = data.getExtras();
                currentImage = (Bitmap) extras.get("data");
                saveImageInGallery(currentImage);
                predict(currentImage); // Trigger prediction after capturing image
            } else {
                handleImageCaptureError();
            }
        } else if (requestCode == GALLERY_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            Uri selectedImageUri = data.getData();
            if (selectedImageUri != null) {
                try {
                    currentImage = resizeImage(MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri));
                    // Removed the predict call here, as it should be triggered when the predict button is clicked
                } catch (IOException e) {
                    e.printStackTrace();
                    handleImageLoadError("Error loading selected image: " + e.getMessage());
                }
            } else {
                handleImageLoadError("Selected image URI is null");
            }
        }
    }

    private void handleImageCaptureError() {
        Toast.makeText(this, "Error capturing image", Toast.LENGTH_SHORT).show();
        Log.e("CameraIntent", "Data or extras are null");
    }

    private Bitmap resizeImage(Bitmap originalImage) {
        int targetWidth = 28;
        int targetHeight = 28;
        return Bitmap.createScaledBitmap(originalImage, targetWidth, targetHeight, true);
    }

    private void handleImageLoadError(String errorMessage) {
        Toast.makeText(this, "Error loading selected image: " + errorMessage, Toast.LENGTH_SHORT).show();
        Log.e("GalleryIntent", "Error loading selected image: " + errorMessage);
    }

    private void saveImageInGallery(Bitmap imageBitmap) {
        String fileName = "my_captured_image.jpg";
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES);

        Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        try {
            if (uri != null) {
                try (OutputStream outputStream = getContentResolver().openOutputStream(uri)) {
                    if (outputStream != null) {
                        imageBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
                        Toast.makeText(this, "Image saved in Pictures", Toast.LENGTH_SHORT).show();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("ImageSaving", "Error saving image: " + e.getMessage());
        }
    }


    private boolean checkPermission() {
        int cameraPermission = ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA);
        int writeExternalStoragePermission = ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE);
        return cameraPermission == PackageManager.PERMISSION_GRANTED && writeExternalStoragePermission == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
        Log.d("Permission", "Requested permissions");
    }
}
