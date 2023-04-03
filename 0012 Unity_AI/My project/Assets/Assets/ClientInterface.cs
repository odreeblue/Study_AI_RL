using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class ClientInterface
{
	private TcpClient socketConnection;
	private static int displayId = 0;
	private PacketManager packetManager = new PacketManager(displayId);

	public void Start()
    {
		ConnectToTcpServer();
	}

	private void ConnectToTcpServer()
	{
		try
		{
			socketConnection = new TcpClient("192.168.200.179", 50001);
		}
		catch (Exception e)
		{
			Debug.Log("On client connect exception " + e);
		}
	}

	public void SendTouch(int x, int y)
    {
		SendMessage(packetManager.GetTouchPacket(x, y));
    }

	public void SendDirection(int direction)
    {
		SendMessage(packetManager.GetDirectionPacket(direction));
    }

	public void SendGaze(int x, int y)
    {
		SendMessage(packetManager.GetGazePacket(x, y));
    }

	public void SendVoice(string message)
	{
		SendMessage(packetManager.GetVoicePacket(message));
	}

	public void SendHandSkeleton(int x, int y)
	{
		SendMessage(packetManager.GetHandSkeletonPacket(x, y));
	}

	/// Send message to server using socket connection. 	
	private void SendMessage(Byte[] buffer)
	{
		if (socketConnection == null)
		{
			return;
		}
		try
		{
			// Get a stream object for writing. 			
			NetworkStream stream = socketConnection.GetStream();
			if (stream.CanWrite)
			{
				// Write byte array to socketConnection stream.                 
				stream.Write(buffer, 0, buffer.Length);
			}
		}
		catch (SocketException socketException)
		{
			Debug.Log("Socket exception: " + socketException);
		}
	}
}
