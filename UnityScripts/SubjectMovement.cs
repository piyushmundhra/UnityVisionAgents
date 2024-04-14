using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

public class SubjectMovement : MonoBehaviour {
    public Tuple<float, float> z_range = new(-3, 3);
    public Tuple<float, float> y_range = new(0,0);
    // public Tuple<float, float> x_range = new(-35, 17);
    public Tuple<float, float> x_range = new(-10, 17);

    public Vector3 startPoint = new(17,0,0);

    public readonly float pathDuration = 70f;

    public List<Vector3> pathPoints;
    public float elapsedTime = 0f;
    public bool isReady = false;

    void Start() {
        transform.position = startPoint;
        transform.LookAt(startPoint + new Vector3(-1,0,0));
        RandomPath();
    }

    void Update() {
        if (!isReady) return;

        elapsedTime += Time.deltaTime;
        float t = elapsedTime / pathDuration;

        if (t < 1f)
        {
            transform.position = GetPosition(t);
        }
    }

    void OnDisable(){
        isReady = false;
        pathPoints.Clear();
        elapsedTime = 0f;
        transform.position = startPoint;
        transform.LookAt(startPoint + new Vector3(-1,0,0));
        RandomPath();
    }

    private void RandomPath(){
        pathPoints.Add(startPoint);
        while(true){
            float randomX = UnityEngine.Random.Range(pathPoints[^1][0] - 1, pathPoints[^1][0]);
            float randomZ = UnityEngine.Random.Range(pathPoints[^1][2] - 1f, pathPoints[^1][2] + 1f);
            randomZ = Mathf.Clamp(randomZ, z_range.Item1, z_range.Item2);

            if (randomX <= x_range.Item1){
                randomX = x_range.Item1;
                pathPoints.Add(new Vector3(randomX, 0, randomZ));
                break;
            } else {
                pathPoints.Add(new Vector3(randomX, 0, randomZ));
            }
        }
    }

    public Vector3 GetPosition(float t) {
        int numSections = pathPoints.Count - 3;
        int currPt = Mathf.Min(Mathf.FloorToInt(t * numSections), numSections - 1);
        float u = t * numSections - currPt;

        Vector3 p0 = pathPoints[currPt];
        Vector3 p1 = pathPoints[currPt + 1];
        Vector3 p2 = pathPoints[currPt + 2];
        Vector3 p3 = pathPoints[currPt + 3];

        return 0.5f * (
            (-p0 + 3f * p1 - 3f * p2 + p3) * u * u * u +
            (2f * p0 - 5f * p1 + 4f * p2 - p3) * u * u +
            (-p0 + p2) * u +
            2f * p1
        );
    }

    public Vector3 GetCurrentPosition(){
        return transform.position;
    }
}
