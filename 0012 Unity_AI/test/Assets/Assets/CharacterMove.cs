using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class CharacterMove : MonoBehaviour
{
    public float speed = 5f;
    private Rigidbody characterRigidbody;
    float inputX, inputZ;

    void Start()
    {
        Debug.Log("Start CharacterMove");
        characterRigidbody = GetComponent<Rigidbody>();

        Server.Instance.SetTouchCallback(new CallbackTouch(OnTouch));
        Server.Instance.SetDirectionCallback(new CallbackDirection(OnDirection));
    }

    void Update()
    {
        Vector3 velocity = new Vector3(inputX, 0, inputZ);
        velocity *= speed;
        characterRigidbody.velocity = velocity;
    }

    void OnDirection(int direction)
    {
        Debug.Log("CharacterMove : " + direction);
        switch (direction)
        {
            case 0:
                inputZ = 1f;
                inputX = 0f;
                break;
            case 1:
                inputZ = -1f;
                inputX = 0f;
                break;
            case 2:
                inputX = 1f;
                inputZ = 0f;
                break;
            case 3:
                inputX = -1f;
                inputZ = 0f;
                break;
        }
    }

    void OnTouch(int x, int y)
    {
        Debug.Log("CharacterMove : " + x + ", " + y);
    }
}