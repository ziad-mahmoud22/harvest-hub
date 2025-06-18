
const express = require('express');
const app = express();
const { Buffer } = require('node:buffer');
const fs = require('fs'); // Import fs module correctly


app.use(express.json({ limit: '10mb' })); 

app.post('/results', (req, res) => {
    try {
        const { image, text } = req.body;
        console.log("Received request body:", req.body); 

        if (!image) {
          console.error("No 'image' field found in request body.");
          return res.status(400).send("Missing 'image' field in request");
        }



        const imageBuffer = Buffer.from(image, 'base64');

        fs.writeFileSync('received_image.jpg', imageBuffer); 
        console.log('Received Text:', text);


        res.json({ message: 'Image and text received successfully', text });

    } catch (error) {
        console.error('Error:', error); 
        res.status(500).send('Error receiving image');
    }
});

app.listen(3000, () => console.log('Express server listening on port 3000'));


