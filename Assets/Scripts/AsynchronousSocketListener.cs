using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityStandardAssets.Characters.FirstPerson;
using WindowsInput;

// State object for reading client data asynchronously  
public class StateObject
{
    // Client  socket.  
    public Socket workSocket = null;
    // Size of receive buffer.  
    public const int BufferSize = 1024;
    // Receive buffer.  
    public byte[] buffer = new byte[BufferSize];
    // Received data string.  
    public StringBuilder sb = new StringBuilder();
}

public class AsynchronousSocketListener : MonoBehaviour
{
    //public BoxCollider rewardTrigger;
    public FirstPersonController FPScontroller;
    public CollisionEventTrigger CollisionEventTrigger;
    public PauseScreen pause;

    public static bool instanceExists = false;

    public enum FrameTransmissionProtocol { Continuous, AsReply, OnRequest }

    private int incomingBufferSize = 1024;
    public int port = 5005;
    public static string endToken = "<EOF>";
    public FrameTransmissionProtocol frame_protocol = FrameTransmissionProtocol.OnRequest;

    private RenderTexture transmissionRenderTexture;
    private Texture2D transmissionTexture;

    private Socket listener;
    private StateObject state;
    private IPEndPoint localEndPoint;
    // Thread signal.  
    public ManualResetEvent allDone = new ManualResetEvent(false);

    public AsynchronousSocketListener() { }

    public void StartListening()
    {
        // Establish the local endpoint for the socket.  
        // The DNS name of the computer  
        // running the listener is "host.contoso.com".  
        IPAddress ipAddress = Dns.GetHostEntry("localhost").AddressList[0];
        localEndPoint = new IPEndPoint(ipAddress, port);

        // Create a TCP/IP socket.  
        listener = new Socket(AddressFamily.InterNetwork,
            SocketType.Stream, ProtocolType.Tcp);

        try
        {
            Debug.Log("Binding to local end point " + localEndPoint.ToString());
            listener.Bind(localEndPoint);
            listener.Listen(100);
        }
        catch (Exception e)
        {
            //Debug.Log(e.ToString());
        }
    }

    public void AcceptCallback(IAsyncResult ar)
    {
        // Get the socket that handles the client request.  
        Socket listener = (Socket)ar.AsyncState;
        Socket handler = listener.EndAccept(ar);

        // Create the state object.  
        state = new StateObject();
        state.workSocket = handler;

        // Signal the main thread to continue.  
        allDone.Set();
    }

    public void ProcessRead()
    {
        String content = String.Empty;

        // Retrieve the state object and the handler socket  
        Socket handler = state.workSocket;
        state.workSocket.Blocking = false;
        //handler.ReceiveTimeout = 10;
        // Read data from the client socket.   
        int bytesRead = 0;
        if (handler.Connected)
            bytesRead = handler.Receive(state.buffer);

        if (bytesRead > 0)
        {
            // There  might be more data, so store the data received so far.  
            state.sb.Append(Encoding.UTF8.GetString(
                state.buffer, 0, bytesRead));

            // Check for end-of-file tag. If it is not there, read   
            // more data.  
            content = state.sb.ToString();
            if (content.IndexOf(endToken) > -1)
            {
                string[] splitContent = content.Split(new string[] { endToken }, StringSplitOptions.RemoveEmptyEntries);
                if (splitContent.Length >= 1)
                {
                    for (int i = 0; i < splitContent.Length; i++)
                    {
                        if(i < splitContent.Length - 1 || content.EndsWith(endToken))
                        {
                            state.sb.Remove(0, splitContent[i].Length + endToken.Length);
                            string tmp = state.sb.ToString();
                            ReceivedData(handler, splitContent[i]);
                        }
                    }
                }
            }
        }
    }

    private void ReceivedData(Socket handler, string content)
    {
        Message m = new Message(content);
        // All the data has been read from the   
        // client. Display it on the console.  
        //Debug.Log("Read " + content.Length + " bytes from socket. \n Data : " + content);
        if (frame_protocol == FrameTransmissionProtocol.AsReply)
        {
            //Debug.Log("Frame Protocol is AsReply, Sending Reply Frame");
            // Echo the data back to the client.  
            SendFrame(handler);
        }
        else if(frame_protocol == FrameTransmissionProtocol.OnRequest)
        {
            
            if(m.Type == "ImageRequest")
            {
                //Debug.Log("Frame Protocol is OnRequest and ImageRequest type received, Sending Reply Frame");
                // Echo the data back to the client.  
                SendFrame(handler);
            }
        }
        Debug.Log("Processing Message Type=" + (m.Type==null?"null":m.Type) + ",Value=" + (m.Value==null?"null":m.Value.ToString()));
        if (m.Type == "Scene")
        {
            SceneManager.LoadScene((int)m.Value);
        }
        else if (m.Type == "Pause")
        {
            //change
            // set humanPlayerMode
            if (m.Value != null)
            {
                pause.pauseMessage = (string)m.Value;
            }
        }
        else if (m.Type == "HumanPlayerMode")
        {
            //change
            // set humanPlayerMode
            if (m.Value != null)
            {
                FPScontroller.setHumanPlayerMode((bool)m.Value);
            }
        }
        else if (m.Type == "PlayerPrefsS")
        {
            string[] vals = (string[])m.Value;
            PlayerPrefs.SetString(vals[0], vals[1]);
        }
        else if (m.Type == "PlayerPrefsF")
        {
            string[] vals = (string[])m.Value;
            PlayerPrefs.SetFloat(vals[0], float.Parse(vals[1]));
        }
        else if (m.Type == "PlayerPrefsI")
        {
            string[] vals = (string[])m.Value;
            PlayerPrefs.SetInt(vals[0], int.Parse(vals[1]));
        }
        else if (m.Type == "Key")
        {
            if (m.Value != null)
            {
                //Execute simulate keystroke
                //WindowsInput.InputSimulator.SimulateKeyDown((VirtualKeyCode)m.Value);

                //change
                string temp = (String)m.Value;
                int direc = temp[0] - '0'; // because char internally is a number, to get its true numeric value, just subtract '0'
                float magni = float.Parse(temp.Substring(1));
                FPScontroller.setTurnDirectionAndMagnitude(direc, magni);
            }
        }
        else if (m.Type == "Reward")
        {
            byte[] typeData = Encoding.UTF8.GetBytes("Reward");
            byte[] EOF = Encoding.UTF8.GetBytes(endToken);
            Send(handler, typeData);
            Send(handler, GetRewardFeedback());
            Send(handler, EOF);
        }
        else if (m.Type == "Score")
        {
            byte[] typeData = Encoding.UTF8.GetBytes("Score");
            byte[] EOF = Encoding.UTF8.GetBytes(endToken);
            Send(handler, typeData);
            Send(handler, GetScoreFeedback());
            Send(handler, EOF);
        }else if (m.Type == "EpisodeEnd")
        {
            byte[] typeData = Encoding.UTF8.GetBytes("EpisodeEnd");
            byte[] EOF = Encoding.UTF8.GetBytes(endToken);
            Send(handler, typeData);
            Send(handler, GetEpisodeEndFeedback());
            Send(handler, EOF);
        }
    }

    public byte[] GetRewardFeedback()
    {
        byte[] reward = Encoding.UTF8.GetBytes((CollisionEventTrigger.collisionReward).ToString());
        return reward;
    }

    public byte[] GetScoreFeedback()
    {
        byte[] score = Encoding.UTF8.GetBytes((CollisionEventTrigger.collisionScore).ToString());
        return score;
    }

    public byte[] GetEpisodeEndFeedback()
    {
        byte[] paused = Encoding.UTF8.GetBytes((pause.pause.ToString()));
        return paused;
    }

    public void SendFrame(Socket handler)
    {
        byte[] typeData = Encoding.UTF8.GetBytes("Image");
        byte[] EOF = Encoding.UTF8.GetBytes(endToken);
        Send(handler, typeData);
        //Send(handler, Encoding.UTF8.GetBytes("ThisIsATestString"));
        Send(handler, GetFrameData());
        Send(handler, EOF);
    }

    public class Message
    {
        private string type;
        private System.Object value;

        public Message()
        {
            Type = null;
            Value = null;
        }

        public Message(string data)
        {
            if (data.StartsWith("ImageRequest"))
            {
                Type = "ImageRequest";
                Value = null;
            }
            else if (data.StartsWith("Scene"))
            {
                Type = "Scene";
                try
                {
                    Value = int.Parse(data.Replace("Scene", "").Replace(AsynchronousSocketListener.endToken, "").Trim());
                }
                catch (Exception)
                {
                    Value = null;
                }
            }
            else if (data.StartsWith("Key"))
            {
                Type = "Key";
                try
                {
                    Value = (VirtualKeyCode)System.Enum.Parse(typeof(VirtualKeyCode), data.Replace("Key", "").Replace(AsynchronousSocketListener.endToken, "").Trim());
                }
                catch (Exception)
                {
                    Value = null;
                }
            }
            else if (data.StartsWith("PlayerPrefsS"))
            {
                Type = "PlayerPrefsS";
                Value = data.Replace("PlayerPrefsS", "").Replace(AsynchronousSocketListener.endToken, "").Split(new char[] { ',' });
            }
            else if (data.StartsWith("PlayerPrefsF"))
            {
                Type = "PlayerPrefsF";
                Value = data.Replace("PlayerPrefsF", "").Replace(AsynchronousSocketListener.endToken, "").Split(new char[] { ',' });
            }
            else if (data.StartsWith("PlayerPrefsI"))
            {
                Type = "PlayerPrefsI";
                Value = data.Replace("PlayerPrefsI", "").Replace(AsynchronousSocketListener.endToken, "").Split(new char[] { ',' });
            }
            else if (data.StartsWith("Reward"))
            {
                Type = "Reward";
                Value = null;
            }
        }

        public string Type
        {
            get
            {
                return type;
            }

            set
            {
                type = value;
            }
        }

        public object Value
        {
            get
            {
                return value;
            }

            set
            {
                this.value = value;
            }
        }
    }

    private byte[] GetFrameData()
    {
        //If the textures are uninitialized or the wrong size, make them and/or destroy and make them
        if(transmissionRenderTexture == null)
            transmissionRenderTexture = new RenderTexture(Camera.main.pixelWidth, Camera.main.pixelHeight, 1);
        if(transmissionTexture == null)
            transmissionTexture = new Texture2D(Camera.main.pixelWidth, Camera.main.pixelHeight);
        if(transmissionRenderTexture.width != Camera.main.pixelWidth || transmissionRenderTexture.height != Camera.main.pixelHeight)
        {
            Destroy(transmissionRenderTexture);
            transmissionRenderTexture = new RenderTexture(Camera.main.pixelWidth, Camera.main.pixelHeight, 1);
        }
        if(transmissionTexture.width != Camera.main.pixelWidth || transmissionTexture.height != Camera.main.pixelHeight)
        {
            Destroy(transmissionTexture);
            transmissionTexture = new Texture2D(Camera.main.pixelWidth, Camera.main.pixelHeight);
        }
        //Get the camera image
        Camera.main.targetTexture = transmissionRenderTexture;
        Camera.main.Render();
        RenderTexture.active = transmissionRenderTexture;
        transmissionTexture.ReadPixels(new Rect(0, 0, transmissionTexture.width, transmissionTexture.height), 0, 0, false);
        Camera.main.targetTexture = null;
        RenderTexture.active = null;
        byte[] bytes = transmissionTexture.EncodeToJPG();
        return bytes;
    }

    private void Send(Socket handler, byte[] data)
    {
        handler.SendTimeout = 10;
        // Convert the string data to byte data using ASCII encoding.  
        handler.Send(data);
    }

    void Awake()
    {
        if (instanceExists)
        {
            Destroy(this.gameObject);
        }
        else
        {
            DontDestroyOnLoad(transform.gameObject);
            instanceExists = true;
            Debug.Log("Starting TCP Server...");
            StartListening();
            currentState = State.Reset;
        }
    }

    void OnApplicationQuit()
    {
        Debug.Log("Application quitting. Shutting down socket.");
        try
        {
            state.workSocket.Shutdown(SocketShutdown.Both);
            state.workSocket.Close();
        }
        catch (SocketException) { }
    }

    private bool firstUpdate = true;
    public enum State { Waiting, Reset, Connected };
    private State currentState;
    void Update()
    { 
        // Bind the socket to the local endpoint and listen for incoming connections.  
        try
        {
            if (currentState == State.Reset) {
                Debug.Log("Entering reset state.");
                // Set the event to nonsignaled state.  
                allDone.Reset();

                // Start an asynchronous socket to listen for connections.  
                Debug.Log("Waiting for a connection...");
                listener.BeginAccept(
                    new AsyncCallback(AcceptCallback),
                    listener);
                currentState = State.Waiting;
            }
            if(currentState == State.Waiting) {
                // Wait until a connection is made before continuing.  
                if (allDone.WaitOne(10))
                    currentState = State.Connected;
            }
            if (currentState == State.Connected)
            {
                if (!IsConnected(listener))
                {
                    Debug.Log("Socket disconnected. Resetting...");
                    currentState = State.Reset;
                }
                else
                {
                    ProcessRead();
                    if (frame_protocol == FrameTransmissionProtocol.Continuous)
                        SendFrame(state.workSocket);
                }
            }
        }
        catch (Exception e)
        {
            //Debug.Log(e.ToString());
        }
    }
    public bool IsConnected(Socket socket)
    {
        try
        {
            return !(socket.Poll(1, SelectMode.SelectRead) && socket.Available == 0);
        }
        catch (SocketException) { return false; }
    }
}