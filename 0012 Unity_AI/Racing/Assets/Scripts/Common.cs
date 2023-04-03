using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class Common : MonoBehaviour
{
    [Range(1,100)]
    public int fFont_Size;
    [Range(0, 1)]
    public float Red, Green, Blue;

    float deltaTime = 0.0f;
    // Start is called before the first frame update
    void Start()
    {
        fFont_Size = fFont_Size == 0 ? 50 : fFont_Size;
    }

    // Update is called once per frame
    void Update()
    {
        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
    }
    private void OnGUI()
    {
        int w = Screen.width, h = Screen.height;

        GUIStyle style = new GUIStyle();

        Rect rect = new Rect(0, 0, w, h * 0.02f);
        style.alignment = TextAnchor.UpperLeft;
        style.fontSize = h * 2 / fFont_Size;
        style.normal.textColor = new Color(Red, Green, Blue, 1.0f);
        float msec = deltaTime * 1000.0f;
        float fps = 1.0f / deltaTime;
        string text = string.Format("{0:0.0} ms ({1:0.} fps)", msec, fps);
        GUI.Label(rect, text, style);
    }
}
[StructLayout(LayoutKind.Sequential,CharSet =CharSet.Ansi)]
public class DataPacket
{
    [MarshalAs(UnmanagedType.R4)]
    public float position_x;
    [MarshalAs(UnmanagedType.R4)]
    public float position_z;
    [MarshalAs(UnmanagedType.R4)]
    public float is_collision;
    [MarshalAs(UnmanagedType.R4)]
    public float image_size;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 51000)]
    //[MarshalAs(UnmanagedType.ByValArray)] --> 이거 안됌 전달 사이즈가 20밖에 안됌
    //[MarshalAs(UnmanagedType.ByValArray, SizeConst = 32000)]
    //public byte[] image;
    public string image;
}