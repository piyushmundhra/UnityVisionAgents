using System.Collections;
using UnityEngine;
using System.Runtime.InteropServices;
using System;
using System.Runtime.ExceptionServices;
using System.IO;
using Unity.MLAgents;
using Unity.Burst;
using Unity.VisualScripting;
using Unity.Barracuda;
using Unity.Mathematics;
using OpenCover.Framework.Model;
using UnityEngine.UIElements;
using System.Collections.Generic;

public static class ClassifierPlugin {
    private static readonly string logFilePath;

    static ClassifierPlugin() {
        string path = Application.persistentDataPath;
        int logFileCount = Directory.GetFiles(path, "logs*.txt").Length;
        logFilePath = Path.Combine(path, $"logs{logFileCount}.txt");
    }

    

    public delegate void LogCallback(string str);

    [DllImport("libagent_lib")]
    public static extern IntPtr create_classifier(string modelFolderPath, string outputFolderPath, LogCallback callback);

    [DllImport("libagent_lib")]
    public static extern void delete_classifier(IntPtr classifier);

    [DllImport("libagent_lib")]
    public static extern void infer_and_store_classifier(IntPtr classifier, byte[][] rgb_frames, float[][] depth_frames, int height, int[] output_array);

    [DllImport("libagent_lib")]
    public static extern void follow_up_classifier(IntPtr classifier, float start_angle, float start_distance, float new_angle, float new_distance);

    [DllImport("libagent_lib")]
    public static extern void bootstrap_train(IntPtr classifier);

    [DllImport("libagent_lib")]
    public static extern void reinforcement_train(IntPtr classifier);

    [DllImport("libagent_lib")]
    public static extern void test_depth_data(IntPtr classifier, float[] depth_data, int width, int height);

    public static void Log(string str){
        using (StreamWriter writer = new(logFilePath, true)){
            writer.WriteLine(str);
        }
        Debug.Log(str);
    }

    public static IntPtr Classifier(string modelFolderPath, string outputFolderPath) {
        return create_classifier(modelFolderPath, outputFolderPath, Log);
    }
}

// attaches to a camera object
public class AgentInterface : MonoBehaviour {
    public GameObject sphere;
    public float optimalDistance = 2f;
    public float maxDistanceDeviation = 3f;
    public float maxAngle = 24f;
    public Vector3 startPoint = new(19,1.5f,0);

    readonly int NUM_INPUT_FRAME_PAIRS = 1;

    public float yawModifier = 1f;
    public float forwardModifier = 0.01f;

    private SubjectMovement subjectMovement;
    private IntPtr classifier;
    public Camera depthCam;
    public Camera rgbCam;
    public Material mat;
    public Shader shader;


    void Start(){
        Screen.SetResolution(640, 640, false);
        ClassifierPlugin.Log(Screen.width + " " + Screen.height);
        subjectMovement = sphere.GetComponent<SubjectMovement>();

        Vector3 spherePosition = subjectMovement.startPoint;
        spherePosition.y = 1.5f;

        string modelFolderPath = System.IO.Path.Combine(Application.dataPath, "Models");
        classifier = ClassifierPlugin.Classifier(modelFolderPath, Application.persistentDataPath);   

        rgbCam = GetComponent<Camera>();;
        rgbCam.fieldOfView = 60;

        rgbCam.transform.position = startPoint;
        rgbCam.transform.LookAt(spherePosition);

        depthCam.fieldOfView = 60;
        depthCam.depthTextureMode = DepthTextureMode.DepthNormals;

        depthCam.transform.position = startPoint;
        depthCam.transform.LookAt(spherePosition);

        mat = new Material(shader);

        subjectMovement.isReady = true;
        ClassifierPlugin.Log("Starting Distance: " + CurrentDistance());
    }

    void OnEnable(){
        Vector3 spherePosition = subjectMovement.startPoint;
        spherePosition.y = 1.5f;
        transform.position = startPoint;
        transform.LookAt(spherePosition);
        subjectMovement.isReady = true;

        ClassifierPlugin.Log("Starting Distance: " + CurrentDistance());
    }

    float CurrentAngle(){
        Vector3 spherePosition = subjectMovement.GetCurrentPosition();
        Vector3 cameraPosition = transform.position;
        Vector3 cameraToSubject = new Vector3(spherePosition.x, 0, spherePosition.z) - new Vector3(cameraPosition.x, 0, cameraPosition.z);

        float angle = Vector3.SignedAngle(new Vector3(transform.forward.x, 0, transform.forward.z), cameraToSubject, Vector3.up);
        // negative angle since facing -x
        return -angle;
    }

    float CurrentDistance(){
        Vector3 spherePosition = subjectMovement.GetCurrentPosition();
        Vector3 cameraPosition = transform.position;
        Vector3 cameraToSubject = new Vector3(spherePosition.x, 0, spherePosition.z) - new Vector3(cameraPosition.x, 0, cameraPosition.z);
        return optimalDistance - cameraToSubject.magnitude;
    }

    float[] GetDepthData(int height, int x, int y, int screenWidth, int screenHeight){
        RenderTexture tempRT = new RenderTexture(screenWidth, screenHeight, 24);
        depthCam.targetTexture = tempRT;
        depthCam.Render();

        RenderTexture.active = tempRT;

        Texture2D depthTexture = new(height, height, TextureFormat.RFloat, false);
        depthTexture.ReadPixels(new Rect(x, y, height, height), 0, 0);
        depthTexture.Apply();

        RenderTexture.active = null;
        depthCam.targetTexture = null;

        Color[] depthColors = depthTexture.GetPixels();
        float[] depthData = new float[depthColors.Length];
        for (int i = 0; i < depthColors.Length; i++)
        {
            depthData[i] = depthColors[i].r;
        }

        return depthData;
    }

    byte[] GetRgbData(int height, int x, int y, int screenWidth, int screenHeight){
        RenderTexture tempRT = new RenderTexture(screenWidth, screenHeight, 24);
        rgbCam.targetTexture = tempRT;
        rgbCam.Render();

        RenderTexture.active = tempRT;

        Texture2D rgbTexture = new(height, height, TextureFormat.RGB24, false);
        rgbTexture.ReadPixels(new Rect(x, y, height, height), 0, 0);
        rgbTexture.Apply();

        RenderTexture.active = null;
        rgbCam.targetTexture = null;

        byte[] rgbData = rgbTexture.GetRawTextureData();

        return rgbData;
    }

    IEnumerator CollectFramesAndInfer(){
        yield return new WaitForEndOfFrame();
        // ClassifierPlugin.Log("\n- - - - - - - - - - - - -\n");
        int screenWidth = Screen.width;
        int screenHeight = Screen.height;

        int height = screenHeight;

        int x = (screenWidth - height) / 2;
        int y = (screenHeight - height) / 2;

        List<float[]> depth_frames = new();
        List<byte[]> rgb_frames = new();
        for (int i = 0; i < NUM_INPUT_FRAME_PAIRS; i++){
            if (i != 0) {yield return new WaitForSeconds(0.1f);}

            float[] depthData = GetDepthData(height, x, y, screenWidth, screenHeight);
            depth_frames.Add(depthData);

            byte[] frame = GetRgbData(height, x, y, screenWidth, screenHeight);
            rgb_frames.Add(frame);
        }

        // Convert lists back to arrays
        float[][] depth_frames_array = depth_frames.ToArray();
        byte[][] rgb_frames_array = rgb_frames.ToArray();
        
        var startingAngle = CurrentAngle();
        var startingDistance = CurrentDistance();

        int[] actions = new int[2];
        ClassifierPlugin.infer_and_store_classifier(classifier, rgb_frames_array, depth_frames_array, height, actions);
        float yaw = actions[0];
        float forward = actions[1];
        ClassifierPlugin.Log("Yaw: " + yaw + ", Forward: " + forward);
        transform.Rotate(0, yaw, 0);
        transform.Translate(0, 0, forward);

        var angle = CurrentAngle();
        var distance = CurrentDistance();
        ClassifierPlugin.follow_up_classifier(classifier, startingAngle, startingDistance, angle, distance);
        ClassifierPlugin.Log("Angle: " + angle + ", Distance: " + distance);
        if (subjectMovement.elapsedTime > 18 || math.abs(angle) > maxAngle || math.abs(distance) > maxDistanceDeviation){
            SubjectMovement subjectMovementScript = sphere.GetComponent<SubjectMovement>();
            FollowSphere followSphereScript = GetComponent<FollowSphere>();
            subjectMovementScript.enabled = false;
            followSphereScript.enabled = false;

            ClassifierPlugin.Log("\n- - - - - - - - - - - - -\n");
            ClassifierPlugin.bootstrap_train(classifier);
            ClassifierPlugin.Log("\n------------------------------------------------------------------------------\n");

            subjectMovementScript.enabled = true;
            followSphereScript.enabled = true;
        } 

    }

    IEnumerator RotateAndWait(){
        var angle = CurrentAngle();
        var distance = CurrentDistance();
        ClassifierPlugin.Log("Angle: " + angle + ", Distance: " + distance);
        yield return new WaitForSeconds(2);
        transform.Rotate(0, 12, 0);
        angle = CurrentAngle();
        distance = CurrentDistance();
        ClassifierPlugin.Log("Angle: " + angle + ", Distance: " + distance);
        yield return new WaitForSeconds(2);
        transform.Rotate(0, -12, 0);
        angle = CurrentAngle();
        distance = CurrentDistance();
        ClassifierPlugin.Log("Angle: " + angle + ", Distance: " + distance);
    }
    void Update(){
        if (subjectMovement != null){
            StartCoroutine(CollectFramesAndInfer());
        }
    }

    void OnDestroy(){
        ClassifierPlugin.delete_classifier(classifier);
    }
}
