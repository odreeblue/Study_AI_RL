using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Runtime.InteropServices;

public delegate void CallbackDirection(int direction, int done);// delegate 선언
public class Server : MonoBehaviour
{
    #region private members
    private TcpListener tcpListener_server;
    //private TcpClient tcpClient_Client;

    private Thread tcpListenerThread;
    
    #endregion
    public DataPacket datapacket;
    
    //public float position_x = 0f;
    //public float position_z = 0f;
    //public float is_collision = 0f;
    public List<float> SendData = null;
    //public byte[] imagedata = null;
    public string imagedata = null;
    //public bool change_position = false;
    public static Server instance = null;
    CallbackDirection callbackDirection; // delegate(대리자)

    public static Server Instance
    {
        get
        {
            if (instance == null)
            {
                return null;
            }
            return instance;
        }
    }

    public void SetDirectionCallback(CallbackDirection callback)// 매개변수 delegate 타입
    {

        // callbackDirection += new CallbackDirection(OnDirection) 
        if (callbackDirection == null)
        {
            callbackDirection = callback;
        }
        else
        {
            callbackDirection += callback;
        }
    }
    private void Awake() // 게임이 시작되기전, 모든 변수와 게임의 상태를 초기화하기 위해서 호출
                         // Start()보다 먼저 호출됨, MonoBehaviour.Awake()
    {
        Debug.Log("Start Server");
        instance = this; // this = Server
        datapacket = new DataPacket();
        
        tcpListenerThread = new Thread(new ThreadStart(ListenForIncommingRequest));
        tcpListenerThread.IsBackground = true;
        tcpListenerThread.Start();
    }
    
    private void ListenForIncommingRequest()
    {

        try
        {
            tcpListener_server = new TcpListener(IPAddress.Parse("127.0.0.1"), 50001);
            tcpListener_server.Start();
            Debug.Log("Server is listening");
            while (true)
            {
                TcpClient tcpClient_Client = tcpListener_server.AcceptTcpClient();
                Thread tcpClient_Action = new Thread(new ParameterizedThreadStart(Action_Client));
                tcpClient_Action.Start(tcpClient_Client);
            }
        }
        catch (SocketException socketException)
        {
            //tcpListener_server.
            Debug.Log("SocketException " + socketException.ToString());
        }
    }
    private void Action_Client(object tcpClient)
    {
        TcpClient client = tcpClient as TcpClient;
        using (NetworkStream stream = client.GetStream()) // Client로 Stream받기
        {
            do
            {
                if (stream.CanRead)
                {
                    Byte[] direction_bytes = new Byte[4];
                    Byte[] isdone = new byte[4];
                    stream.Read(direction_bytes, 0, 4);//데이터 읽기
                    stream.Read(isdone, 0, 4);
                    int direction = BitConverter.ToInt32(direction_bytes, 0);//byte -> int 로 변환
                    int done = BitConverter.ToInt32(isdone, 0);
                    callbackDirection(direction, done);// 받은 action signal --> sphere 전달
                                                       // Sphere의 OnDirection 호출
                    while (true)
                    {

                        if (SendData.Count == 4 && imagedata != null) // Sphere로부터 데이터가 다 입력 됐으면
                        {
                            datapacket.position_x = SendData[0];
                            datapacket.position_z = SendData[1];
                            datapacket.is_collision = SendData[2];
                            datapacket.image_size = SendData[3];
                            datapacket.image = imagedata;
                            break;
                        }
                        else
                        {
                            continue;
                        }

                    }
                    SendData.Clear();
                    imagedata = null;
                    byte[] buffer = new byte[Marshal.SizeOf(datapacket)];
                    unsafe
                    {
                        fixed (byte* fixed_buffer = buffer)
                        {
                            Marshal.StructureToPtr(datapacket, (IntPtr)fixed_buffer, false);
                        }
                    }

                    stream.Write(buffer, 0, Marshal.SizeOf(datapacket));
                    Debug.Log("보낸 데이터 크기는 : " + Marshal.SizeOf(datapacket));
                    stream.Flush();
                }
                

            } while (true);
        }
    }
    
}


