using UnityEngine;
using System.Collections;
using UnityStandardAssets.ImageEffects;
using UnityEngine.UI;
using UnityStandardAssets.Characters.FirstPerson;
using UnityEngine.SceneManagement;

public class PauseScreen : MonoBehaviour {

    public string pauseMessage;

    public ColorCorrectionCurves screenEffect;
    public FirstPersonController controller;
    public Canvas overlay;
    public Text displayText;
    public bool pause;
    public string pauseText;
    public KeyCode pauseKey;
    public bool resetOnKey;

    public void SetPauseMessage(string message)
    {
        pauseMessage = message;
    }

    // Update is called once per frame
    void RefreshPauseState () {
        screenEffect.enabled = pause;
        overlay.gameObject.SetActive(pause);
        controller.enabled = !pause;
        displayText.text = PauseText;
	}

    void Update()
    {
        if (Input.GetKeyUp(pauseKey) || pauseMessage == "Space") //change
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
