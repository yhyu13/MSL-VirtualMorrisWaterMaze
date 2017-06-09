using UnityEngine;
using System.Collections;
using UnityEngine.SceneManagement;

public class TrialTimer : MonoBehaviour {
    public TrialConfigurationLoader configLoader;
    public PauseScreen pause;
    private float startTime;
    private float pauseTime;
    private float prevTime;
    public float trialTime;
    private bool done;
    public float elapsed;
	// Use this for initialization
	void Start () {
        startTime = Time.time;
        done = false;
        pauseTime = 0f;
        elapsed = 0f;
	}
	
	// Update is called once per frame
	void Update () {
        if (!done)
        {
            float currentTime = Time.time;
            if (pause.Pause)
                pauseTime += currentTime - prevTime;

            elapsed = (currentTime - startTime) - pauseTime;
            if (elapsed > trialTime)
                EndTrial();

            prevTime = currentTime;
        }
	}

    public void EndTrial()
    {
        pause.pauseText = "End of Trial\r\nPress Space To Continue";
        pause.Pause = true;
        pause.pauseKey = KeyCode.Space;
        done = true;
        configLoader.NextIteration();
        pause.resetOnKey = true;

        // go back to the menu once a trial is over
        //SceneManager.LoadScene("Menu");
    }
}
