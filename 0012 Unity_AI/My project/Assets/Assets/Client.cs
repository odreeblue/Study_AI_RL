using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class Client : MonoBehaviour
{
	private ClientInterface clientInterface;

	// Use this for initialization 	
	void Start()
	{
		clientInterface = new ClientInterface();
		clientInterface.Start();
	}
	// Update is called once per frame
	void Update()
	{
		if (Input.GetKeyDown(KeyCode.UpArrow))
			clientInterface.SendDirection(0);
		if (Input.GetKeyDown(KeyCode.DownArrow))
			clientInterface.SendDirection(1);
		if (Input.GetKeyDown(KeyCode.RightArrow))
			clientInterface.SendDirection(2);
		if (Input.GetKeyDown(KeyCode.LeftArrow))
			clientInterface.SendDirection(3);
		if (Input.GetKeyDown(KeyCode.Space))
			clientInterface.SendVoice("Hello Unity");
	}

}