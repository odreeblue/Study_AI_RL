using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;


public delegate void CallbackTouch(int x, int y);
public delegate void CallbackDirection(int direction);
public delegate void CallbackGaze(int x, int y);
public delegate void CallbackVoice(string message);
public delegate void CallbackHandSkeleton(int x, int y);

public class Server : MonoBehaviour
{
    #region private members
    private TcpListener tcpListener;
    private Thread tcpListenerThread;
    private TcpClient connectedTcpClient;
    #endregion

    private static Server instance = null;
    CallbackTouch callbackTouch;
    CallbackDirection callbackDirection;
    CallbackGaze callbackGaze;
    CallbackVoice callbackVoice;
    CallbackHandSkeleton callbackHandSkeleton;

    void Awake()
    {
        Debug.Log("Start Server");
        instance = this;

        // Start TcpServer background thread
        tcpListenerThread = new Thread(new ThreadStart(ListenForIncommingRequest));
        tcpListenerThread.IsBackground = true;
        tcpListenerThread.Start();
    }

    // Update is called once per frame
    void Update()
    {

    }

    public static Server Instance
    {
        get
        {
            if(instance == null)
            {
                return null;
            }
            return instance;
        }
    }

    public void SetTouchCallback(CallbackTouch callback)
    {
        if(callbackTouch == null)
        {
            callbackTouch = callback;
        } else
        {
            callbackTouch += callback;
        }
    }

    public void SetDirectionCallback(CallbackDirection callback)
    {
        if(callbackDirection == null)
        {
            callbackDirection = callback;
        } else
        {
            callbackDirection += callback;
        }
    }

    public void SetGazeCallback(CallbackGaze callback)
    {
        if(callbackGaze == null)
        {
            callbackGaze = callback;
        } else
        {
            callbackGaze += callback;
        }
    }

    public void SetVoiceCallback(CallbackVoice callback)
    {
        if(callbackVoice == null)
        {
            callbackVoice = callback;
        } else
        {
            callbackVoice += callback;
        }
    }

    public void SetHandSkeletonCallback(CallbackHandSkeleton callback)
    {
        if(callbackHandSkeleton == null)
        {
            callbackHandSkeleton = callback;
        } else
        {
            callbackHandSkeleton += callback;
        }
    }

    // Runs in background TcpServerThread; Handles incomming TcpClient requests
    private void ListenForIncommingRequest()
    {
        try
        {
            // Create listener on 192.168.0.2 port 50001
            tcpListener = new TcpListener(IPAddress.Parse("192.168.200.179"), 50001);
            tcpListener.Start();
            Debug.Log("Server is listening");

            while (true)
            {
                using (connectedTcpClient = tcpListener.AcceptTcpClient())
                {
                    // Get a stream object for reading
                    using (NetworkStream stream = connectedTcpClient.GetStream())
                    {
                        // Read incomming stream into byte array.
                        do
                        {
                            Byte[] bytesTypeOfService = new Byte[4];
                            Byte[] bytesDisplayId = new Byte[4];
                            Byte[] bytesPayloadLength = new Byte[4];

                            int lengthTypeOfService = stream.Read(bytesTypeOfService, 0, 4);
                            int lengthDisplayId = stream.Read(bytesDisplayId, 0, 4);
                            int lengthPayloadLength = stream.Read(bytesPayloadLength, 0, 4);
                            Debug.Log("lengthTypeOfService : "+lengthTypeOfService);
                            Debug.Log("Array.Reverse(lengthTypeOfService)"+Array.Reverse(bytesTypeOfService));
                            Debug.Log("lengthDisplayId : "+lengthDisplayId);
                            Debug.Log("lengthPayloadLength : "+lengthPayloadLength);
                            
                            if (lengthTypeOfService <= 0 && lengthDisplayId <= 0 && lengthPayloadLength <= 0)
                            {
                                break;
                            }

                            // Reverse byte order, in case of big endian architecture
                            if (!BitConverter.IsLittleEndian)
                            {
                                Array.Reverse(bytesTypeOfService);
                                Array.Reverse(bytesDisplayId);
                                Array.Reverse(bytesPayloadLength);
                            }

                            int typeOfService = BitConverter.ToInt32(bytesTypeOfService, 0);
                            int displayId = BitConverter.ToInt32(bytesDisplayId, 0);
                            int payloadLength = BitConverter.ToInt32(bytesPayloadLength, 0);

                            if (typeOfService == 3)
                            {
                                payloadLength = 1012;
                            }

                            Byte[] bytes = new Byte[payloadLength];
                            int length = stream.Read(bytes, 0, payloadLength);

                            HandleIncommingRequest(typeOfService, displayId, payloadLength, bytes);
                        } while (true);
                    }
                }
            }
        }
        catch (SocketException socketException)
        {
            Debug.Log("SocketException " + socketException.ToString());
        }
    }

    // Handle incomming request
    private void HandleIncommingRequest(int typeOfService, int displayId, int payloadLength, byte[] bytes)
    {
        Debug.Log("=========================================");
        Debug.Log("Type of Service : " + typeOfService);
        Debug.Log("Display Id      : " + displayId);
        Debug.Log("Payload Length  : " + payloadLength);
        switch (typeOfService)
        {
            case 0:
                TouchHandler(displayId, payloadLength, bytes);
                break;
            case 1:
                DirectionHander(displayId, payloadLength, bytes);
                break;
            case 2:
                GazeHandler(displayId, payloadLength, bytes);
                break;
            case 3:
                VoiceHandler(displayId, payloadLength, bytes);
                break;
            case 4:
                BodySkeletonHandler(displayId, payloadLength, bytes);
                break;
            case 5:
                HandSkeletonHandler(displayId, payloadLength, bytes);
                break;
        }
    }

    // Handle Touch Signal
    private void TouchHandler(int displayId, int payloadLength, byte[] bytes)
    {
        Debug.Log("Execute Touch Handler");
        int x_axis = BitConverter.ToInt32(bytes, 0);
        int y_axis = BitConverter.ToInt32(bytes, 4);
        Debug.Log("X axis     : " + x_axis);
        Debug.Log("Y axis     : " + y_axis);
        if(callbackTouch != null)
        {
            callbackTouch(x_axis, y_axis);
        }
    }

    // Handle Direction Signal
    private void DirectionHander(int displayId, int payloadLength, byte[] bytes)
    {
        Debug.Log("Execute Direction Handler");
        int direction = BitConverter.ToInt32(bytes, 0);
        Debug.Log("Direction  : " + direction);
        if(callbackDirection != null)
        {
            callbackDirection(direction);
        }
        
    }

    // Handle Gaze Signal
    private void GazeHandler(int displayId, int payloadLength, byte[] bytes)
    {
        Debug.Log("Execute Gaze Handler");
        int x_axis = BitConverter.ToInt32(bytes, 0);
        int y_axis = BitConverter.ToInt32(bytes, 4);
        Debug.Log("X axis     : " + x_axis);
        Debug.Log("Y axis     : " + y_axis);
        if(callbackGaze != null)
        {
            callbackGaze(x_axis, y_axis);
        }
    }

    // Handle Voice Signal
    private void VoiceHandler(int displayId, int payloadLength, byte[] bytes)
    {
        Debug.Log("Execute Voice Handler");
        string str = Encoding.Default.GetString(bytes);
        Debug.Log("Text       : " + str);
        if(callbackVoice != null)
        {
            callbackVoice(str);
        }
    }

    // Handle Body Skeleton Signal
    private void BodySkeletonHandler(int displayId, int payloadLength, byte[] bytes)
    {
        Debug.Log("Execute Body Skeleton Handler");
        // TODO
    }

    // Handle Hand Skeleton Signal
    private void HandSkeletonHandler(int displayId, int payloadLength, byte[] bytes)
    {
        Debug.Log("Execute Hand Skeleton Handler");
        int x_axis = BitConverter.ToInt32(bytes, 0);
        int y_axis = BitConverter.ToInt32(bytes, 4);
        Debug.Log("X axis     : " + x_axis);
        Debug.Log("Y axis     : " + y_axis);
        if(callbackHandSkeleton != null)
        {
            callbackHandSkeleton(x_axis, y_axis);
        }
    }
}

