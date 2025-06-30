//======================================== Including the libraries.
#include "esp_camera.h"
#include "soc/rtc_cntl_reg.h"
#include "soc/soc.h"
#include <ArduinoJson.h>
#include <HTTPClient.h>
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
#define FLASH_LED_PIN 4
//const int Sensor_PIN = 34; // Commented out as ADC is not needed

//======================================== Insert your network credentials.
const char *ssid = "";
const char *password = "";
const char *zone = "zone_1";
char *model_name = "tomato";
String serverName = "192.168.1.10";
String serverPath = "/predict";
const int serverPort = 8001;

//======================================== Variables for Timer/Millis.
unsigned long previousMillis = 0;
const long Interval = 20000; // Photo capture every 20 seconds
volatile bool sendNow = true;
//volatile int ADC_Reading = 0; // Commented out as ADC is not needed

//======================================== WiFi & Server Clients
WiFiClient client;
WiFiServer commandServer(9000);
String device_id;

//======================================== Device Registration
void registerWithServer() {
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("Registering device with server...");
    HTTPClient http;
    String url = "http://" + serverName + ":8001/register";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    String jsonData = "{\"device_id\": \"" + device_id + "\", \"ip\": \"" +
                      WiFi.localIP().toString() + "\", \"zone\": \"" + zone +
                      "\"}";
    int httpResponseCode = http.POST(jsonData);
    if (httpResponseCode > 0) {
      Serial.printf("Registration successful. Status code: %d\n",
                    httpResponseCode);
    } else {
      Serial.printf("Registration failed. Error: %s\n",
                    http.errorToString(httpResponseCode).c_str());
    }
    http.end();
  } else {
    Serial.println("WiFi not connected. Skipping registration.");
  }
}

//======================================== Function to send photo to server.
String sendPhotoToServer() {
  sendNow = false;
  String responseBody = "";

  Serial.println("\n-----------");
  Serial.println("Preparing to capture image...");

  camera_fb_t *fb = NULL;
  for (int attempt = 0; attempt < 3; attempt++) {
    digitalWrite(FLASH_LED_PIN, HIGH);
    delay(150); // Small delay for flash to stabilize
    fb = esp_camera_fb_get();
    digitalWrite(FLASH_LED_PIN, LOW);
    if (fb) {
      Serial.println("Image captured successfully");
      break;
    }
    Serial.println("Capture failed, retrying...");
    delay(200);
  }

  if (!fb) {
    Serial.println("Camera capture failed after 3 attempts. Restarting...");
    delay(1000);
    ESP.restart();
    return "ERROR";
  }

  /* // Commented out ADC-related code
  int currentADC_Reading = ADC_Reading;
  Serial.println("Current sensor value to be sent: " +
                 String(currentADC_Reading));
  */

  Serial.println("Connecting to server: " + serverName);
  if (client.connect(serverName.c_str(), serverPort)) {
    Serial.println("Connection successful!");

    String boundary = "----Boundary" + String(millis());
    String header = "--" + boundary + "\r\n";
    header += "Content-Disposition: form-data; name=\"device_id\"\r\n\r\n";
    header += device_id + "\r\n";

    header += "--" + boundary + "\r\n";
    header += "Content-Disposition: form-data; name=\"model_name\"\r\n\r\n";
    header += String(model_name) + "\r\n";

    /* // Commented out ADC-related form data
    header += "--" + boundary + "\r\n";
    header += "Content-Disposition: form-data; name=\"sensor_value\"\r\n\r\n";
    header += String(currentADC_Reading) + "\r\n";
    */

    header += "--" + boundary + "\r\n";
    header += "Content-Disposition: form-data; name=\"image\"; "
              "filename=\"capture.jpg\"\r\n";
    header += "Content-Type: image/jpeg\r\n\r\n";

    String footer = "\r\n--" + boundary + "--\r\n";
    uint32_t contentLength = header.length() + fb->len + footer.length();

    client.println("POST " + serverPath + " HTTP/1.1");
    client.println("Host: " + serverName);
    client.println("Content-Length: " + String(contentLength));
    client.println("Content-Type: multipart/form-data; boundary=" + boundary);
    client.println("Connection: close");
    client.println();
    client.print(header);

    client.write(fb->buf, fb->len);

    client.print(footer);
    esp_camera_fb_return(fb);
    Serial.println("Image sent to server");

    // Robust HTTP response parsing
    unsigned long timeout = millis() + 10000;
    while (client.connected() && !client.available() && millis() < timeout) {
      vTaskDelay(pdMS_TO_TICKS(10)); // Yield while waiting
    }

    // Skip HTTP headers
    while (client.available()) {
      String line = client.readStringUntil('\n');
      if (line == "\r") {
        Serial.println("Headers received.");
        break; // Blank line indicates end of headers
      }
    }

    // Read the body
    if (client.available()) {
      responseBody = client.readString();
    }

    client.stop();
    Serial.println("Server response: " + responseBody);

    if (responseBody.length() > 0) {
      StaticJsonDocument<256> doc;
      DeserializationError error = deserializeJson(doc, responseBody);
      if (!error && doc.containsKey("command")) {
        return doc["command"].as<String>();
      }
    }
  } else {
    Serial.println("Connection to server failed");
    esp_camera_fb_return(fb); // Ensure frame buffer is returned on failure
  }
  return "ERROR";
}

//======================================== TASK 1: Command Listener
void commandListenerTask(void *pvParameters) {
  Serial.println("Command listener task started on core " +
                 String(xPortGetCoreID()));
  for (;;) {
    WiFiClient cmdClient = commandServer.available();
    if (cmdClient) {
      Serial.println("Command client connected");
      unsigned long timeout = millis() + 5000;
      String command = "";
      while (cmdClient.connected() && millis() < timeout) {
        if (cmdClient.available()) {
          command = cmdClient.readStringUntil('\n');
          command.trim();
          break;
        }
        vTaskDelay(pdMS_TO_TICKS(10));
      }

      if (command.length() > 0) {
        Serial.println("Received command: " + command);
        if (command == "TAKE_PHOTO") {
          sendNow = true;
          cmdClient.println("ACK:TAKE_PHOTO");
        } else {
          cmdClient.println("ERROR:UNKNOWN_COMMAND");
        }
      } else {
        cmdClient.println("ERROR:TIMEOUT");
      }
      cmdClient.stop();
      Serial.println("Command client disconnected");
    }
    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

/* // Commented out TASK 2: ADC Reader as it is not needed
void adcReadTask(void *pvParameters) {
  Serial.println("ADC read task started on core " + StringxPortGetCoreID()));
  for (;;) {
    ADC_Reading = analogRead(Sensor_PIN);
    vTaskDelay(pdMS_TO_TICKS(500));
  }
}
*/

//======================================== Setup function
void setup() {
  device_id = WiFi.macAddress();
  Serial.begin(115200);
  Serial.println("\n\nStarting ESP32-CAM Device");
  Serial.println("Device ID: " + device_id);

  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  WiFi.mode(WIFI_STA);
  Serial.println("Connecting to: " + String(ssid));
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("\nWiFi connected! IP: " + WiFi.localIP().toString());

  registerWithServer();
  commandServer.begin();
  Serial.println("Command server started on port 9000");

  xTaskCreatePinnedToCore(commandListenerTask, "CommandListener", 4096, NULL, 1,
                          NULL, 0);
  /* // Commented out ADC task creation
  xTaskCreatePinnedToCore(adcReadTask, "ADCReader", 2048, NULL, 1, NULL, 0);
  */

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
  config.frame_size = FRAMESIZE_UXGA;
  config.jpeg_quality = 10;
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    ESP.restart();
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_brightness(s, 1);
  s->set_saturation(s, 0);

  Serial.println("Camera initialized successfully. System ready.");
}

//======================================== Main Loop (runs on Core 1)
void loop() {
  if (millis() - previousMillis >= Interval || sendNow) {
    previousMillis = millis();
    String command = sendPhotoToServer();
    if (command == "REGISTER_AGAIN") {
      Serial.println("Re-registering device...");
      registerWithServer();
    }
  }
  delay(100);
}