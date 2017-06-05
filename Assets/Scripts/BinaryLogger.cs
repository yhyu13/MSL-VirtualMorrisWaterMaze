using UnityEngine;
using System.Collections;
using System.IO;
using System;
using System.Collections.Generic;

public class BinaryLogger : MonoBehaviour {
	
    public TrialConfigurationLoader configLoader; 
    public int numberOfExecutionsForThisTrial; 
    public int recordEveryWhatPercent;
    private int iterations; 
    private bool doRecord;

    public string dateTimeFormat = "yyyy-MM-dd_HH-mm-ss";
    public string filenameFormat = "<subid>_<trial>_<iteration>_<datetime>.dat";
    private string header = "";
    private static int headerLength = 1024;

    public Camera cam;

    public List<KeyCode> keys;
    public List<string> buttons;

    private BinaryWriter writer;

    private bool firstUpdate = true;

	// Use this for initialization
	void Start () {
        string filename = filenameFormat;

        if (PlayerPrefs.HasKey("subid"))
            filename = filename.Replace("<subid>", PlayerPrefs.GetString("subid"));
        else
            filename = filename.Replace("<subid>", "unk");
        if(PlayerPrefs.HasKey("trial"))
            filename = filename.Replace("<trial>", "" + PlayerPrefs.GetInt("trial"));
        else
            filename = filename.Replace("<trial>", "u");
        if (PlayerPrefs.HasKey("iteration"))
            filename = filename.Replace("<iteration>", "" + PlayerPrefs.GetInt("iteration"));
        else
            filename = filename.Replace("<iteration>", "u");

        DateTime time = DateTime.Now;
        string timeString = time.ToString(dateTimeFormat);
        filename = filename.Replace("<datetime>", timeString);
		
	doRecord = iterations % (numberOfExecutionsForThisTrial * recordEveryWhatPercent / 100) == 0;
        if (!doRecord) // change
        {
            return;
        }

        Stream stream = new StreamWriter(filename).BaseStream;
        writer = new BinaryWriter(stream);
	}
	
	// Update is called once per frame
	void Update () {
	
	if (!doRecord)
        {
            return;
        }
	
        if (firstUpdate)
        {
            
            header = "vmwm-version,1,time,f,timeScale,f,posXYZ,fff,rotXYZW,ffff,";
            for (int i = 0; i < keys.Count; i++)
                header += "key" + keys[i].ToString() + "_" + i.ToString().PadLeft(2, '0') + ",b,";
            for (int i = 0; i < buttons.Count; i++)
                header += "button" + buttons[i].ToString() + "_" + i.ToString().PadLeft(2, '0') + ",b,";

            if (header.Length > headerLength)
            {
                Debug.LogError("Error: Header input length is longer than the 1k Maximum.");
                Application.Quit();
            }

            if (header.Length < headerLength)
                header = header.PadRight(headerLength);

            writer.Write(header);

            firstUpdate = false;
        }
        writer.Write(DateTime.Now.ToBinary());
        writer.Write(Time.time);
        writer.Write(cam.transform.position.x);
        writer.Write(cam.transform.position.y);
        writer.Write(cam.transform.position.z);
        writer.Write(cam.transform.rotation.x);
        writer.Write(cam.transform.rotation.y);
        writer.Write(cam.transform.rotation.z);
        writer.Write(cam.transform.rotation.w);
        for(int i = 0; i < keys.Count; i++)
            writer.Write(Input.GetKey(keys[i]));
        for (int i = 0; i < buttons.Count; i++)
        {
            bool state = false;
            try { state = Input.GetButton(buttons[i]); } catch (ArgumentException) { }
            writer.Write(state);
        }
    }

    void OnApplicationQuit()
    {
	if (!doRecord)
        {
            return;
        }
        writer.Close();
    }

    void OnDisable()
    {
	if (!doRecord)
        {
            return;
        }
        writer.Close();
    }
}
