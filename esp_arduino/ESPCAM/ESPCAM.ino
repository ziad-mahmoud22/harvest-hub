// ESP32_CAM_Send_Photo_to_Server_with_ID
//======================================== Including the libraries.
#include "esp_camera.h"
#include "soc/rtc_cntl_reg.h"
#include "soc/soc.h"
#include <WiFi.h>

//======================================== CAMERA_MODEL_AI_THINKER GPIO.
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// LED Flash PIN (GPIO 4)
#define FLASH_LED_PIN 4

//======================================== Insert your network credentials.
const char *ssid = "Moh";
const char *password = "123456789";

//======================================== Variables for Timer/Millis.
unsigned long previousMillis = 0;
const int Interval = 20000; // Photo capture every 20 seconds.

// Server Address or Server IP.
String serverName = "192.168.43.91"; // Change this to your server's IP
String serverPath = "/predict";
const int serverPort = 8001;

// Variable to set capture photo with LED Flash.
bool LED_Flash_ON = true;
bool sendNow = true;

// Initialize WiFiClient.
WiFiClient client;
WiFiServer commandServer(9000);  // Listen for commands on port 9000

// Unique device ID (MAC address)
String device_id;

//======================================== Function to send photo to server.
bool sendPhotoToServer() 
{
  sendNow = false;
  String AllData;
  String DataBody;

  Serial.println();
  Serial.println("-----------");
  Serial.println("Taking a photo...");

  if (LED_Flash_ON) {
    digitalWrite(FLASH_LED_PIN, HIGH);
    delay(1000);
  }

  for (int i = 0; i <= 3; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) 
    {
      Serial.println("Camera capture failed");
      Serial.println("Restarting the ESP32 CAM.");
      delay(1000);
      ESP.restart();
      return false;
    }
    esp_camera_fb_return(fb);
    delay(200);
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    Serial.println("Restarting the ESP32 CAM.");
    delay(1000);
    ESP.restart();
    return false;
  }

  if (LED_Flash_ON) digitalWrite(FLASH_LED_PIN, LOW);

  Serial.println("Taking a photo was successful.");

  String model_name = "tomato";

  Serial.println("Connecting to server: " + serverName);

  if (client.connect(serverName.c_str(), serverPort)) {
    Serial.println("Connection successful!");

    // Include device_id in the multipart form data
    String head = "--dataMarker\r\nContent-Disposition: form-data; name=\"device_id\"\r\n\r\n" + device_id + "\r\n--dataMarker\r\nContent-Disposition: form-data; name=\"model_name\"\r\n\r\n" + model_name + "\r\n--dataMarker\r\nContent-Disposition: form-data; name=\"image\"; filename=\"ESP32CAMCap.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
    String boundary = "\r\n--dataMarker--\r\n";

    uint32_t imageLen = fb->len;
    uint32_t dataLen = head.length() + boundary.length();
    uint32_t totalLen = imageLen + dataLen;

    client.println("POST " + serverPath + " HTTP/1.1");
    client.println("Host: " + serverName);
    client.println("Content-Length: " + String(totalLen));
    client.println("Content-Type: multipart/form-data; boundary=dataMarker");
    client.println();
    client.print(head);

    uint8_t *fbBuf = fb->buf;
    size_t fbLen = fb->len;
    for (size_t n = 0; n < fbLen; n += 1024) {
      if (n + 1024 < fbLen) {
        client.write(fbBuf, 1024);
        fbBuf += 1024;
      } else if (fbLen % 1024 > 0) {
        size_t remainder = fbLen % 1024;
        client.write(fbBuf, remainder);
      }
    }
    client.print(boundary);

    esp_camera_fb_return(fb);

    int timoutTimer = 10000;
    long startTimer = millis();
    boolean state = false;
    Serial.println("Response : ");
    while ((startTimer + timoutTimer) > millis()) 
    {
      Serial.print(".");
      delay(200);
      while (client.available()) 
      {
        char c = client.read();
        if (c == '\n') {
          if (AllData.length() == 0) state = true;
          AllData = "";
        } else if (c != '\r') {
          AllData += String(c);
        }
        if (state) DataBody += String(c);
        startTimer = millis();
      }
      if (DataBody.length() > 0) break;
    }
    client.stop();
    Serial.println(DataBody);
    Serial.println("-----------");
    return DataBody.indexOf("TAKE_PHOTO_NOW") != -1;
  } else {
    client.stop();
    Serial.println("Connection to " + serverName + " failed.");
    Serial.println("-----------");
    return false;
  }
}

//======================================== Setup function.
void setup() 
{
  
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Disable brownout detector
  Serial.begin(115200);
  Serial.println();

  pinMode(FLASH_LED_PIN, OUTPUT);
  WiFi.mode(WIFI_STA);

  Serial.print("Connecting to : ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  int connecting_process_timed_out = 20; // 20 seconds
  connecting_process_timed_out *= 2;
  
  while (WiFi.status() != WL_CONNECTED) 
  {
    Serial.print(".");
    delay(500);
    if (connecting_process_timed_out-- <= 0) {
      Serial.println();
      Serial.println("Failed to connect to " + String(ssid));
      Serial.println("Restarting the ESP32 CAM.");
      delay(1000);
      ESP.restart();
    }
  }

  Serial.println();
  Serial.println("Successfully connected to ");
  Serial.println(ssid);

  commandServer.begin();
  Serial.println("Command server started on port 9000.");


  // Set the device ID using the MAC address
  device_id = WiFi.macAddress();
  Serial.println("Device ID: " + device_id);

  // Camera configuration
  Serial.println("Set the camera ESP32 CAM...");
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) 
  {
    config.frame_size = FRAMESIZE_QSXGA;
    config.jpeg_quality = 10; // 0-63, lower means higher quality
    config.fb_count = 2;
  } else 
  {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 8;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    Serial.println("Restarting the ESP32 CAM.");
    delay(1000);
    ESP.restart();
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_UXGA);

  Serial.println("Set camera ESP32 CAM successfully.");
  Serial.println("ESP32-CAM captures and sends photos every 20 seconds.");
}



void checkForClientCommands() 
{
  
  WiFiClient incomingClient = commandServer.accept();
  
  if (incomingClient) 
  {
    Serial.println("Client connected to command server.");
    String command = "";
    while (incomingClient.connected()) 
    {
      while (incomingClient.available()) 
      {
        char c = incomingClient.read();
        if (c == '\n') 
        {
          command.trim();
          Serial.println( "Received command: " + command );

          // Put your commands handling HERE
          if ( command == "TAKE_PHOTO" ) 
          {
            sendNow = true;
            incomingClient.println( "OK" );
          } else 
          {
            incomingClient.println("UNKNOWN_COMMAND");
          }

          incomingClient.stop();
          return;
        } else {
          command += c;
        }
      }
    }
  }
}



//======================================== Loop function.
void loop() 
{
   checkForClientCommands();  // Check for incoming commands

  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= Interval || sendNow == true ) 
  {
    previousMillis = currentMillis;
    bool takeAnother = sendPhotoToServer();
    while (takeAnother) 
    {
      takeAnother = sendPhotoToServer();
    }
  }
  
}
