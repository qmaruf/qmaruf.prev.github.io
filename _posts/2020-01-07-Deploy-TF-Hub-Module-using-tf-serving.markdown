---
title: Deploy Tf Hub Module Using Tf Serving
date: 2020-01-07 00:00:00 Z
---

I am working on pet project to compare text similarity. As a goto methold, I want to get an embedding 
corresponding to each provided text. In the next step, I will calculate the similarity between these embedding to find most
similar texts to a provided one.

Let's use universal encoder from tensorflow hub to extract embedddings for each text. As I want to deploy the service 
on the cloud, I will use tf serving to serve the model. I've used the following code to serve the model using docker. I'll 
explore later how to deploy on the cloud.

```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import tensorflow_hub as hub


export_dir = "./universal_encoder/00000001"
with tf.Session(graph=tf.Graph()) as sess:
    module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2") 
    input_params = module.get_input_info_dict()
    text_input = tf.placeholder(name='text', 
        dtype=input_params['text'].dtype, 
        shape=input_params['text'].get_shape())
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embeddings = module(text_input)
    tf.saved_model.simple_save(sess, 
        export_dir, 
        inputs={'text': text_input}, 
        outputs={'embeddings': embeddings}, 
        legacy_init_op=tf.tables_initializer())
```

This code block is adapted from https://stackoverflow.com/questions/50788080/how-to-make-the-tensorflow-hub-embeddings-servable-using-tensorflow-serving
It will export the hub module into `./universal_encoder` directory using version `1`.

To serve the model using docker install docker first and pull tensorflow serving image using the following command.
```
docker pull tensorflow/serving  
docker run -p 8501:8501 -v absolute_path_to/universal_encoder:/models/universal_encoder -e MODEL_NAME=universal_encoder -t tensorflow/serving
```
At this point, the tf embedding module will be up and running. We can use `curl` to check the response the module. Use the following 
command in the terminal.

```
curl http://localhost:8501/v1/models/universal_encoder
```
The response should be something like this.
```
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}
```

Yeeee. It's working. We can check the model meta data using
```
curl http://localhost:8501/v1/models/universal_encoder/metadata
```

Output
```
{
"model_spec":{
 "name": "universal_encoder",
 "signature_name": "",
 "version": "1"
}
,
"metadata": {"signature_def": {
 "signature_def": {
  "serving_default": {
   "inputs": {
    "text": {
     "dtype": "DT_STRING",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "text:0"
    }
   },
   "outputs": {
    "embeddings": {
     "dtype": "DT_FLOAT",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "512",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "module_apply_default/Encoder_en/hidden_layers/l2_normalize:0"
    }
   },
   "method_name": "tensorflow/serving/predict"
  }
 }
}
}
}
```

Seems like we can use the `inputs` endpoint to send request to the model. It will send the response using `output`. Let's try 
the following command.
```
curl -d '{"inputs": ["hello world"]}' -X POST http://localhost:8501/v1/models/universal_encoder:predict
```
We are asking the model to send embeddings for the text `hello world`. The response is
```
{
    "outputs": [
        [
            -0.037585672,
            -0.0128515968,
            -0.0215193778,
            0.0627601296,
            -0.0105388267,
            ....
        ]
    ]
}
```






