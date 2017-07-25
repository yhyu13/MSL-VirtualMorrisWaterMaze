using UnityEngine;
using System.Collections;
using System;
using System.Xml.Serialization;
using System.IO;
using System.Collections.Generic;

[System.Serializable]
public class Configuration : System.Object {

    public static Configuration Deserialize(string xml)
    {
        Configuration obj = new Configuration();
        XmlSerializer serializer = new XmlSerializer(typeof(Configuration));
        using(TextReader textReader = new StringReader(xml))
        {
            obj = (Configuration)serializer.Deserialize(textReader);
        }
        return obj;
    }

    public static string Serialize(Configuration c)
    {
        XmlSerializer serializer = new XmlSerializer(typeof(Configuration));
        string result = "";
        using(TextWriter textWriter = new StringWriter())
        {
            serializer.Serialize(textWriter, c);
            result = textWriter.ToString();
        }
        return result;
    }

    public Configuration()
    {
        TrialStrings = "Practice - Flags,Practice - Hills,Practice - Visible Platform,Trial 1,Trial 2-5,Trial 6-10,Trial 11-15,Probe Trial".Split(new char[] { ',' });

        int n = TrialStrings.Length;

        InitializeDataArrays(n);

        enableTCP = false;
        port = 5005;
        frameMode = 1;
    }

    public Configuration(string path) : base()
    {
        //Initialize the parser
        INIParser parser = new INIParser();
        StreamReader reader = new StreamReader(path);
        string fileContents = reader.ReadToEnd();
        reader.Close();
        parser.OpenFromString(fileContents);
        try
        {
            //Read the trial strings
            string trialStr = parser.ReadValue("Global", "TrialStrings", "Practice - Flags,Practice - Hills,Practice - Visible Platform,Trial 1,Trial 2-5,Trial 6-10,Trial 11-15,Probe Trial");
            TrialStrings = trialStr.Split(new char[] { ',' });

            //Read the global settings

            enableTCP = parser.ReadValue("Global", "EnableTCP", 0) != 0;
            port = parser.ReadValue("Global", "Port", 5005);
            frameMode = parser.ReadValue("Global", "FrameMode", 1);

            //RandomSamplePositions
            string samplePosStr = parser.ReadValue("Global", "RandomSamplePoints", "");
            List<Vector2> randomPositions = new List<Vector2>();
            string[] splt = samplePosStr.Split(new char[] { ',' });
            for (int i = 0; i < splt.Length - 1; i += 2)
                randomPositions.Add(new Vector2(float.Parse(splt[i]), float.Parse(splt[i + 1])));

            //RandomSampleOrientations
            List<Quaternion> randomOrientations = new List<Quaternion>();
            string sampleOriStr = parser.ReadValue("Global", "RandomSampleOrientations", "");
            string [] splt2 = sampleOriStr.Split(new char[] { ',' });
            for (int i = 0; i < splt2.Length; i++)
                randomOrientations.Add(Quaternion.Euler(0f, float.Parse(splt2[i]), 0f));

            //Popualte the data arrays with defaults
            int n = TrialStrings.Length;

            InitializeDataArrays(n);

            //Iterate through the sections
            for(int i = 0; i < TrialStrings.Length; i++) {
                string trialString = TrialStrings[i];

                bool startNewRandomSample = parser.ReadValue(trialString, "StartNewRandomSample", 0) != 0;
                string randomSampleIdx = parser.ReadValue(trialString, "RandomSampleIndex", "");
                int[] randomSampleIndices;
                if (randomSampleIdx != "")
                {
                    string[] rsplit = randomSampleIdx.Split(new char[] { ',' });
                    randomSampleIndices = new int[rsplit.Length];
                    for (int j = 0; j < rsplit.Length; j++)
                        randomSampleIndices[j] = int.Parse(rsplit[j]);
                }
                else
                {
                    randomSampleIndices = new int[0];
                }

                //Get the number of executions to look for values for
                NumberOfExecutions[i] = parser.ReadValue(trialString, "NumberOfExecutions", 1);
                if (NumberOfExecutions[i] < 1) NumberOfExecutions[i] = 1; //Must be at least 1
                if (NumberOfExecutions[i] > 1) //If there are multiple executions, initalize the position/orientation arrays appropriately
                {
                    platformPositions[i] = new Vector2[NumberOfExecutions[i]];
                    playerStartPositions[i] = new Vector2[NumberOfExecutions[i]];
                    playerStartOrientations[i] = new Quaternion[NumberOfExecutions[i]];
                    for(int j = 0; j < NumberOfExecutions[i]; j++)
                    {
                        platformPositions[i][j] = Vector2.zero;
                        playerStartPositions[i][j] = Vector2.zero;
                        playerStartOrientations[i][j] = Quaternion.identity;
                    }
                }

                //Populate the platform positions if specified (zeros default)
                string platformPositionStr = parser.ReadValue(trialString, "PlatformPosition", "");
                if (platformPositionStr != "")
                {
                    string[] split = platformPositionStr.Split(new char[] { ',' });
                    for (int j = 0; j < Math.Floor((float)split.Length / 2f) * 2f; j += 2)
                        if (j < NumberOfExecutions[i] * 2)
                            platformPositions[i][j / 2] = ParseVector2(split[j] + ',' + split[j + 1]);
                }

                //Populate the player start position if specified (zeros default)
                string playerStartPositionStr = parser.ReadValue(trialString, "PlayerStartPosition", "");
                if (playerStartPositionStr != "")
                {
                    string[] split = playerStartPositionStr.Split(new char[] { ',' });
                    for (int j = 0; j < Math.Floor((float)split.Length / 2f) * 2f; j += 2)
                        if (j < NumberOfExecutions[i] * 2)
                            playerStartPositions[i][j / 2] = ParseVector2(split[j] + ',' + split[j + 1]);
                }
                else
                {
                    //Sample from playerprefs and randomsaplepoints
                    int perm = getSamplePlayerPrefsPerm(startNewRandomSample);
                    int[] idxs = permutation5(perm);
                    for (int j = 0; j < NumberOfExecutions[i]; j++)
                        if (j < NumberOfExecutions[i] && j < randomSampleIndices.Length)
                            playerStartPositions[i][j] = randomPositions[idxs[randomSampleIndices[j]]];
                }

                //Populate the player starting orientation if specified
                string orientationStr = parser.ReadValue(trialString, "PlayerStartOrientationRadians", "");
                if (playerStartPositionStr != "")
                {
                    string[] split = playerStartPositionStr.Split(new char[] { ',' });
                    for (int j = 0; j < split.Length; j++)
                        if (j < NumberOfExecutions[i])
                        {
                            try
                            {
                                float orientation = float.Parse(split[j].Trim());
                                playerStartOrientations[i][j] = Quaternion.Euler(0f, orientation, 0f);
                            }
                            catch (Exception) { }
                        }
                }
                else
                {
                    //Sample from playerprefs and randomsampleorientations
                    int perm = getSamplePlayerPrefsPerm(false);
                    int[] idxs = permutation5(perm);
                    for (int j = 0; j < NumberOfExecutions[i]; j++)
                        if (j < NumberOfExecutions[i] && j < randomSampleIndices.Length)
                            playerStartOrientations[i][j] = randomOrientations[idxs[randomSampleIndices[j]]];
                }

                //Read the timeout (float.PositiveInfinity is default)
                string timeoutString = parser.ReadValue(trialString, "TimeoutInMilliseconds", "");
                float timeout = float.PositiveInfinity;
                if (timeoutString.Trim().ToLower() != "none") 
                    timeout = (float)parser.ReadValue(trialString, "TimeoutInMilliseconds", float.PositiveInfinity) / 1000f;
                trialTimeLimits[i] = timeout;

                //Read the visibility booleans
                landmarkVisibilities[i] = parser.ReadValue(trialString, "LandmarkVisibilities", 1) != 0;
                platformVisibilities[i] = parser.ReadValue(trialString, "PlatformVisible", 1) != 0;
                flagVisibilities[i] = parser.ReadValue(trialString, "FlagsVisible", 1) != 0;
                hillVisibilities[i] = parser.ReadValue(trialString, "HillsVisible", 1) != 0;
                platformTriggerEnabled[i] = parser.ReadValue(trialString, "PlatformTriggerEnabled", 1) != 0;

                //Get the sound setting
                SoundEffectsEnabled[i] = parser.ReadValue(trialString, "SoundEffectsEnabled", 1) != 0;

                //Get the movement speed and ensure it is greater than or equal to 0
                MovementSpeeds[i] = (float)parser.ReadValue(trialString, "MovementSpeed", 5.0f);
                if (MovementSpeeds[i] < 0f) MovementSpeeds[i] = 0f;
            }
        }
        catch (Exception) { }
        parser.Close();
    }

    public Vector2 ParseVector2(string vString)
    {
        string[] split = vString.Split(new char[] { ',' });
        try
        {
            string xStr = split[0].Trim().ToLower();
            string yStr = split[0].Trim().ToLower();

            float x = float.Parse(split[0]);
            float y = float.Parse(split[1]);

            Vector2 v = new Vector2(x, y);
            return v;
        }
        catch (Exception) { }

        return new Vector2(float.PositiveInfinity, float.PositiveInfinity);
    }

    private void InitializeDataArrays(int n)
    {
        platformPositions = new Vector2[n][];

        playerStartPositions = new Vector2[n][];
        playerStartOrientations = new Quaternion[n][];

        trialTimeLimits = new float[n];

        landmarkVisibilities = new bool[n];
        platformVisibilities = new bool[n];
        flagVisibilities = new bool[n];
        hillVisibilities = new bool[n];
        platformTriggerEnabled = new bool[n];

        NumberOfExecutions = new int[n];
        SoundEffectsEnabled = new bool[n];
        MovementSpeeds = new float[n];

        for (int i = 0; i < n; i++)
        {
            platformPositions[i] = new Vector2[] { Vector2.zero };
            playerStartPositions[i] = new Vector2[] { Vector2.zero };
            playerStartOrientations[i] = new Quaternion[] { Quaternion.identity };
            trialTimeLimits[i] = -1.0f;
            landmarkVisibilities[i] = true;
            platformVisibilities[i] = false;
            flagVisibilities[i] = false;
            hillVisibilities[i] = false;
            platformTriggerEnabled[i] = true;
            NumberOfExecutions[i] = 1;
            SoundEffectsEnabled[i] = true;
            MovementSpeeds[i] = 5f;
        }
    }

    private bool enableTCP;
    private int port;
    private int frameMode;

    private string[] trialStrings;

    private Vector2[][] platformPositions;

    private Vector2[][] playerStartPositions;
    private Quaternion[][] playerStartOrientations;

    private float[] trialTimeLimits;

    private bool[] landmarkVisibilities;
    private bool[] platformVisibilities;
    private bool[] flagVisibilities;
    private bool[] hillVisibilities;
    private bool[] platformTriggerEnabled;

    private int[] numberOfExecutions;
    private bool[] soundEffectsEnabled;
    private float[] movementSpeeds;

    public string[] TrialStrings
    {
        get
        {
            return trialStrings;
        }
        set
        {
            trialStrings = value;
        }
    }

    public Vector2[][] PlatformPositions
    {
        get
        {
            return platformPositions;
        }

        set
        {
            platformPositions = value;
        }
    }

    public Vector2[][] PlayerStartPositions
    {
        get
        {
            return playerStartPositions;
        }

        set
        {
            playerStartPositions = value;
        }
    }

    public Quaternion[][] PlayerStartOrientations
    {
        get
        {
            return playerStartOrientations;
        }

        set
        {
            playerStartOrientations = value;
        }
    }

    public float[] TrialTimeLimits
    {
        get
        {
            return trialTimeLimits;
        }

        set
        {
            trialTimeLimits = value;
        }
    }

    public bool[] LandmarkVisibilities
    {
        get
        {
            return landmarkVisibilities;
        }

        set
        {
            landmarkVisibilities = value;
        }
    }

    public bool[] PlatformVisibilities
    {
        get
        {
            return platformVisibilities;
        }

        set
        {
            platformVisibilities = value;
        }
    }

    public bool[] FlagVisibilities
    {
        get
        {
            return flagVisibilities;
        }

        set
        {
            flagVisibilities = value;
        }
    }

    public bool[] HillVisibilities
    {
        get
        {
            return hillVisibilities;
        }

        set
        {
            hillVisibilities = value;
        }
    }

    public bool[] PlatformTriggerEnabled
    {
        get
        {
            return platformTriggerEnabled;
        }

        set
        {
            platformTriggerEnabled = value;
        }
    }

    public int[] NumberOfExecutions
    {
        get
        {
            return numberOfExecutions;
        }

        set
        {
            numberOfExecutions = value;
        }
    }

    public bool[] SoundEffectsEnabled
    {
        get
        {
            return soundEffectsEnabled;
        }

        set
        {
            soundEffectsEnabled = value;
        }
    }

    public float[] MovementSpeeds
    {
        get
        {
            return movementSpeeds;
        }

        set
        {
            movementSpeeds = value;
        }
    }

    public bool EnableTCP
    {
        get
        {
            return enableTCP;
        }

        set
        {
            enableTCP = value;
        }
    }

    public int Port
    {
        get
        {
            return port;
        }

        set
        {
            port = value;
        }
    }

    public int FrameMode
    {
        get
        {
            return frameMode;
        }

        set
        {
            frameMode = value;
        }
    }

    public int getSamplePlayerPrefsPerm(bool advance)
    {
        if (!PlayerPrefs.HasKey("sample_perm"))
            PlayerPrefs.SetInt("sample_perm", 0);
        else if (advance)
            PlayerPrefs.SetInt("sample_perm", PlayerPrefs.GetInt("sample_perm", 0) + 1);

        return PlayerPrefs.GetInt("sample_perm") % 120; // mod 5 factorial // magic numbers
    }

    private static int[] permutation5(int idx)
    {
        int permNumber = 5; // magic numbers
        int[] lut = new int[] { 11, 38, 88, 64, 49, 81, 2, 87, 107, 30, 48, 17, 7, 110, 23, 98, 32, 89, 43, 46, 18, 100, 53, 39, 47, 84, 113, 41, 58, 97, 33, 85, 4, 35, 31, 71, 26, 117, 83, 68, 9, 55, 8, 50, 66, 78, 94, 24, 79, 70, 76, 57, 60, 69, 62, 0, 111, 109, 74, 6, 5, 108, 45, 3, 25, 102, 95, 21, 115, 27, 63, 86, 29, 75, 44, 13, 105, 101, 56, 65, 73, 14, 93, 103, 77, 15, 54, 10, 51, 42, 59, 20, 106, 112, 72, 91, 104, 19, 90, 116, 114, 36, 16, 40, 1, 34, 118, 82, 52, 99, 61, 67, 119, 12, 37, 28, 92, 96, 80, 22 };
        int[,] perms = new int[,] { { 0, 1, 2, 3, 4 }, { 0, 1, 2, 4, 3 }, { 0, 1, 3, 2, 4 }, { 0, 1, 3, 4, 2 }, { 0, 1, 4, 2, 3 }, { 0, 1, 4, 3, 2 }, { 0, 2, 1, 3, 4 }, { 0, 2, 1, 4, 3 }, { 0, 2, 3, 1, 4 }, { 0, 2, 3, 4, 1 }, { 0, 2, 4, 1, 3 }, { 0, 2, 4, 3, 1 }, { 0, 3, 1, 2, 4 }, { 0, 3, 1, 4, 2 }, { 0, 3, 2, 1, 4 }, { 0, 3, 2, 4, 1 }, { 0, 3, 4, 1, 2 }, { 0, 3, 4, 2, 1 }, { 0, 4, 1, 2, 3 }, { 0, 4, 1, 3, 2 }, { 0, 4, 2, 1, 3 }, { 0, 4, 2, 3, 1 }, { 0, 4, 3, 1, 2 }, { 0, 4, 3, 2, 1 }, { 1, 0, 2, 3, 4 }, { 1, 0, 2, 4, 3 }, { 1, 0, 3, 2, 4 }, { 1, 0, 3, 4, 2 }, { 1, 0, 4, 2, 3 }, { 1, 0, 4, 3, 2 }, { 1, 2, 0, 3, 4 }, { 1, 2, 0, 4, 3 }, { 1, 2, 3, 0, 4 }, { 1, 2, 3, 4, 0 }, { 1, 2, 4, 0, 3 }, { 1, 2, 4, 3, 0 }, { 1, 3, 0, 2, 4 }, { 1, 3, 0, 4, 2 }, { 1, 3, 2, 0, 4 }, { 1, 3, 2, 4, 0 }, { 1, 3, 4, 0, 2 }, { 1, 3, 4, 2, 0 }, { 1, 4, 0, 2, 3 }, { 1, 4, 0, 3, 2 }, { 1, 4, 2, 0, 3 }, { 1, 4, 2, 3, 0 }, { 1, 4, 3, 0, 2 }, { 1, 4, 3, 2, 0 }, { 2, 0, 1, 3, 4 }, { 2, 0, 1, 4, 3 }, { 2, 0, 3, 1, 4 }, { 2, 0, 3, 4, 1 }, { 2, 0, 4, 1, 3 }, { 2, 0, 4, 3, 1 }, { 2, 1, 0, 3, 4 }, { 2, 1, 0, 4, 3 }, { 2, 1, 3, 0, 4 }, { 2, 1, 3, 4, 0 }, { 2, 1, 4, 0, 3 }, { 2, 1, 4, 3, 0 }, { 2, 3, 0, 1, 4 }, { 2, 3, 0, 4, 1 }, { 2, 3, 1, 0, 4 }, { 2, 3, 1, 4, 0 }, { 2, 3, 4, 0, 1 }, { 2, 3, 4, 1, 0 }, { 2, 4, 0, 1, 3 }, { 2, 4, 0, 3, 1 }, { 2, 4, 1, 0, 3 }, { 2, 4, 1, 3, 0 }, { 2, 4, 3, 0, 1 }, { 2, 4, 3, 1, 0 }, { 3, 0, 1, 2, 4 }, { 3, 0, 1, 4, 2 }, { 3, 0, 2, 1, 4 }, { 3, 0, 2, 4, 1 }, { 3, 0, 4, 1, 2 }, { 3, 0, 4, 2, 1 }, { 3, 1, 0, 2, 4 }, { 3, 1, 0, 4, 2 }, { 3, 1, 2, 0, 4 }, { 3, 1, 2, 4, 0 }, { 3, 1, 4, 0, 2 }, { 3, 1, 4, 2, 0 }, { 3, 2, 0, 1, 4 }, { 3, 2, 0, 4, 1 }, { 3, 2, 1, 0, 4 }, { 3, 2, 1, 4, 0 }, { 3, 2, 4, 0, 1 }, { 3, 2, 4, 1, 0 }, { 3, 4, 0, 1, 2 }, { 3, 4, 0, 2, 1 }, { 3, 4, 1, 0, 2 }, { 3, 4, 1, 2, 0 }, { 3, 4, 2, 0, 1 }, { 3, 4, 2, 1, 0 }, { 4, 0, 1, 2, 3 }, { 4, 0, 1, 3, 2 }, { 4, 0, 2, 1, 3 }, { 4, 0, 2, 3, 1 }, { 4, 0, 3, 1, 2 }, { 4, 0, 3, 2, 1 }, { 4, 1, 0, 2, 3 }, { 4, 1, 0, 3, 2 }, { 4, 1, 2, 0, 3 }, { 4, 1, 2, 3, 0 }, { 4, 1, 3, 0, 2 }, { 4, 1, 3, 2, 0 }, { 4, 2, 0, 1, 3 }, { 4, 2, 0, 3, 1 }, { 4, 2, 1, 0, 3 }, { 4, 2, 1, 3, 0 }, { 4, 2, 3, 0, 1 }, { 4, 2, 3, 1, 0 }, { 4, 3, 0, 1, 2 }, { 4, 3, 0, 2, 1 }, { 4, 3, 1, 0, 2 }, { 4, 3, 1, 2, 0 }, { 4, 3, 2, 0, 1 }, { 4, 3, 2, 1, 0 } };
        int[] output = new int[permNumber];
        int lutIdx = lut[idx];
        for (int i = 0; i < output.Length; i++)
            output[i] = perms[lutIdx, i];
        return output;
    }
}
