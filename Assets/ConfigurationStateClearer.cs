using UnityEngine;
using System.Collections;

public class ConfigurationStateClearer : MonoBehaviour {

    public static bool exists = false;

    public bool firstLoad = true;

	// Use this for initialization
	void Start () {
        if (exists)
        {
            DestroyImmediate(this.gameObject);
            return;
        }
        exists = true;
        DontDestroyOnLoad(this.gameObject);
    }
}
