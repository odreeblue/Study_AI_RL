using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using System.Linq;
using System.Net.Sockets;
using System.Text;

public class Capture : MonoBehaviour
{
    public Camera capture_camera;
    private int resWidth;
    private int resHeight;
    string path;
    public static Capture instance = null;
    public static Capture Instance
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
    // Start is called before the first frame update
    void Awake()
    {
        capture_camera = GetComponent<Camera>();
        
        instance = this;
        resWidth = Screen.width;
        resHeight = Screen.height;
        Debug.Log("width : " + resWidth + "  height : " + resHeight);
        path = Application.dataPath + "/ScreenShot/";
        Debug.Log(path);
    }

    // Update is called once per frame
    //public byte[] ScreenShot()
    public string ScreenShot()
    {
        
        DirectoryInfo dir = new DirectoryInfo(path);
        if (!dir.Exists)
        {
            Directory.CreateDirectory(path);
        }
        string name;
        name = path + System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss") + ".png";
        //name = path  + "currentfile.png";
        RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
        capture_camera.targetTexture = rt;
        RenderTexture.active = rt;

        Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);

        //Texture2D screenShot = new Texture2D(658, 658, TextureFormat.RGB24, false);
        //Texture2D screenShot = new Texture2D(84, 84, TextureFormat.RGB24, false);
        Rect rec = new Rect(0, 0, screenShot.width, screenShot.height);
        Debug.Log("screenShot.width : " + screenShot.width + " screenShot.height : " + screenShot.height);
        //Rect rec = new Rect(300, 0, 658, 658);
        //Rect rec = new Rect(300, 0, 84, 84);
        capture_camera.Render();

        
        //screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
        screenShot.ReadPixels(rec, 0, 0);
        screenShot.Apply();
        
        capture_camera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
        //screenShot.Enc
        //screenShot.
        //byte[] bytes = screenShot.Get;
        //byte[] bytes = screenShot.EncodeToPNG();
        byte[] bytes = screenShot.EncodeToJPG();
        //byte[] serverMessageAsByteArray = Encoding.ASCII.GetBytes(bytes.ToString());
        //Debug.Log(bytes);
        //Debug.Log("bytes : "+ bytes[0..20]);
        string file = Convert.ToBase64String(bytes);
        //byte[] tempbytes = Convert.FromBase64String(file);
        //string printstring = file[20];
        //Debug.Log("string file [0:20] : " + file[0:20]);
        Debug.Log("Bytes Size is :"+bytes.Length);
        //Debug.Log("serverMessageAsByteArray Size is :" + serverMessageAsByteArray.Length);
        Debug.Log("string Size is :" + file.Length);
        //Debug.Log("tempbytes Size is :" + tempbytes.Length);
        Destroy(screenShot);
        //File.WriteAllText(name, file);
        File.WriteAllBytes(name, bytes);
        return file;
        


    }
    
}
