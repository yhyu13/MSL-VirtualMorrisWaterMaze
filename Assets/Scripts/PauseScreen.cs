using UnityEngine;
using System.Collections;
using UnityStandardAssets.ImageEffects;
using UnityEngine.UI;
using UnityStandardAssets.Characters.FirstPerson;
using UnityEngine.SceneManagement;

public class PauseScreen : MonoBehaviour {

    public string pauseMessage = "null";

    public ColorCorrectionCurves screenEffect;
    //public AugmentedController controller;
    public Canvas overlay;
    public Text displayText;
    public bool pause = false;
    public string pauseText;
    public KeyCode pauseKey;
    public bool resetOnKey = false;

    public void SetPauseMessage(string message)
    {
        pauseMessage = message;
    }

    // Update is called once per frame
    void RefreshPauseState () {
        screenEffect.enabled = pause;
        overlay.gameObject.SetActive(pause);
        //controller.enabled = !pause;
        displayText.text = PauseText;
	}

    void Update()
    {
        if (pauseMessage == "Space")
        {
            if (resetOnKey)
            {
                Debug.Log("Restarting Scene");
                CameraFade.SetScreenOverlayColor(Color.black);
                SceneManager.LoadScene(SceneManager.GetActiveScene().name);
            }
        }

        if (Input.GetKeyUp(pauseKey)) //change
        {
            if (resetOnKey)
            {
                Debug.Log("Restarting Scene");
                CameraFade.SetScreenOverlayColor(Color.black);
                SceneManager.LoadScene(SceneManager.GetActiveScene().name);
            }
            else
            {
                pause = !pause;
                RefreshPauseState();
            }
        }
    }

    public bool Pause
    {
        get { return pause; }
        set { pause = value; RefreshPauseState(); }
    }

    public string PauseText
    {
        get
        {
            return pauseText;
        }

        set
        {
            pauseText = value;
        }
    }
}
