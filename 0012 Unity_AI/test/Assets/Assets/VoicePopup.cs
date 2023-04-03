using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VoicePopup : MonoBehaviour
{
    NoticeMessage notice;
    bool IsMessage = false;
    string message = "";
    // Start is called before the first frame update
    void Start()
    {
        notice = FindObjectOfType<NoticeMessage>();
        Server.Instance.SetVoiceCallback(new CallbackVoice(OnVoice));
    }

    // Update is called once per frame
    void Update()
    {
        if(IsMessage == true)
        {
            notice.Print(message);
            IsMessage = false;
        }
    }

    void OnVoice(string msg)
    {
        Debug.Log("OnVoice : " + msg);
        IsMessage = true;
        message = msg;
    }
}
