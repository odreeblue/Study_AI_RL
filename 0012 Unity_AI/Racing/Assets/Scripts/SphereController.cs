using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Net.Sockets;
using UnityEngine.SceneManagement;
//using UnityEditor.TextCore.Text;
using UnityEngine.XR;
//using System.Drawing;

//public delegate void CallbackPosAndCol(float x, float y, int isCol);// delegate 선언
public class SphereController : MonoBehaviour
{
    public float speed = 5.0f;
    public Rigidbody SphereRigidbody;
    public bool collision_flag ; // 충돌했는지 확인하는 플래
    public Vector3 move;
    public Vector3 SubjectPosition;
    public Vector3 SpherePosition;
    public int game_done;
    public GameObject bonus1,bonus2,bonus3,bonus4,bonus5;
    public bool IsGoal;
    public float bonus;
    //public Vector3 LastSpherePosition;
    //public Vector3 reflect;
    void Start()
    {
        bonus1 = GameObject.Find("Bonus_2,0");
        bonus2 = GameObject.Find("Bonus_1,4");
        bonus3 = GameObject.Find("Bonus_0,0");
        bonus4 = GameObject.Find("Bonus_-2,0");
        bonus5 = GameObject.Find("Bonus_Goal");
        Time.timeScale = speed;
        Debug.Log("Start CharacterMove");

        SphereRigidbody = GetComponent<Rigidbody>();
        
        collision_flag = false;
       
        SpherePosition = SphereRigidbody.transform.position;
        
        speed = 1.0f;

        move = new Vector3(0, 0, 0);
        
        Server.Instance.SetDirectionCallback(new CallbackDirection(OnDirection)); //static Server 반환됨
        IsGoal = false;
    }
    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.name.Substring(0,3) == "Out" || col.gameObject.name.Substring(0,3) == "InS")
        {
            collision_flag = true;
            Debug.Log(col.gameObject.name);
        }
        if (col.gameObject.name.Substring(0, 1) == "B")
        {
            col.transform.position = new Vector3(col.transform.position.x, -2.0f,col.transform.position.z);
            bonus = 10.0f;
            if (col.gameObject.name == "Bonus_Goal")
            {
                IsGoal = true;
                bonus = 15.0f;
            }

        }
        //
    }

    // Update is called once per frame
    void Update()
    {
        // Python에서 게임을 종료하라는 신호가 들어왔을 시에
        if (game_done != 0)
        {
            move = new Vector3(0, 0, 0);
            SphereRigidbody.transform.position = new Vector3(4.0f, 2.0f, -4.0f);
            SpherePosition = new Vector3(4.0f, 0.0f, -4.0f);
            
            Server.Instance.SendData.Add(SphereRigidbody.transform.position.x); //python 에 전달되었을 때 의미 없지만 통신을 위해 
            Server.Instance.SendData.Add(SphereRigidbody.transform.position.z); //python 에 전달되었을 때 의미 없지만 통신을 위해 
            Server.Instance.SendData.Add(1.0f);  //python 에 전달되었을 때 의미 없지만 통신을 위해 
        }
        //SubjectPosition = SpherePosition + move;
        game_done = 0;
        //LastSpherePosition = SphereRigidbody.transform.position;
        if (move != new Vector3(0,0,0))// move가 입력이되었고 게임이 끝나지 않았으면
        {
            SphereRigidbody.transform.Translate(move * speed * Time.deltaTime);
        }
        //reflect = (LastSpherePosition - SpherePosition) * 5.0f ;

    }
    void LateUpdate()
    {
        if(move != new Vector3(0, 0, 0))
        {
            float epsilon = 0.1f;
            float loss_x = SphereRigidbody.transform.position.x - SubjectPosition.x;
            //float loss_y = SphereRigidbody.transform.position.y - SubjectPosition.y;
            float loss_z = SphereRigidbody.transform.position.z - SubjectPosition.z;

            float pos_y = SphereRigidbody.transform.position.y;

            float abs_xz = Math.Abs(loss_x) + Math.Abs(loss_z);
            Debug.Log("SphereRigidbody.x: " + SphereRigidbody.transform.position.x + "-SubjectPosition.x :" + SubjectPosition.x + " ,loss x : " + loss_x);
            Debug.Log("SphereRigidbody.z: " + SphereRigidbody.transform.position.z + "-SubjectPosition.z :" + SubjectPosition.z + " ,loss z : " + loss_z);
            //Debug.Log("loss z : " + loss_z);
            Debug.Log("total abs loss : " + abs_xz + "  Collision_flag " + collision_flag + "  move : " + move[0] + "," + move[1] + "," + move[2]);
            // 목표 도달 & 충돌 없음
            if ((abs_xz <= epsilon) && collision_flag == false && move != new Vector3(0, 0, 0))// && pos_y > -0.1)
            {
                Server.Instance.SendData.Add(SphereRigidbody.transform.position.x);
                Server.Instance.SendData.Add(SphereRigidbody.transform.position.z);
                if (IsGoal == true) // 목표에 도달했다면
                {

                    Server.Instance.SendData.Add(bonus);
                    string Image_data = Capture.Instance.ScreenShot();
                    byte[] Length = Convert.FromBase64String(Image_data);
                    
                    Debug.Log("Image_data Size is : " + Image_data.Length);
                    Server.Instance.imagedata = Image_data;
                    Server.Instance.SendData.Add(Image_data.Length);


                    SphereRigidbody.transform.position = new Vector3(4.0f, 3.0f, -4.0f);
                    SpherePosition = new Vector3(4.0f, 0.0f, -4.0f);
                    bonus1.transform.position = new Vector3(bonus1.transform.position.x, 0.6f, bonus1.transform.position.z);
                    bonus2.transform.position = new Vector3(bonus2.transform.position.x, 0.6f, bonus2.transform.position.z);
                    bonus3.transform.position = new Vector3(bonus3.transform.position.x, 0.6f, bonus3.transform.position.z);
                    bonus4.transform.position = new Vector3(bonus4.transform.position.x, 0.6f, bonus4.transform.position.z);
                    bonus5.transform.position = new Vector3(bonus5.transform.position.x, 0.6f, bonus5.transform.position.z);
                    IsGoal = false;
                    
                }
                else // 목표에 도달하지 않았다1
                {
                    Server.Instance.SendData.Add(bonus);
                    //byte[] Image_data = Capture.Instance.ScreenShot();
                    string Image_data = Capture.Instance.ScreenShot();
                    
                    byte[] Length = Convert.FromBase64String(Image_data);
                    
                    Debug.Log("Image_data Size is : " + Image_data.Length);
                    Server.Instance.imagedata = Image_data;
                    Server.Instance.SendData.Add(Image_data.Length);

                    SpherePosition = SphereRigidbody.transform.position;
                    
                }
                bonus = 0.0f;
                move = new Vector3(0, 0, 0);

            }
            // 충돌 있음
            else if (collision_flag == true && move != new Vector3(0, 0, 0))// && pos_y > -0.1)
            {
                //SphereRigidbody.transform.position = SphereRigidbody.transform.position+reflect;
                Server.Instance.SendData.Add(SphereRigidbody.transform.position.x);
                Server.Instance.SendData.Add(SphereRigidbody.transform.position.z);
                Server.Instance.SendData.Add(1.0f);
                string Image_data = Capture.Instance.ScreenShot();
                byte[] Length = Convert.FromBase64String(Image_data);
                
                Debug.Log("Image_data Size is : " + Image_data.Length);
                Server.Instance.imagedata = Image_data;
                Server.Instance.SendData.Add(Image_data.Length);


                move = new Vector3(0, 0, 0);
                SphereRigidbody.transform.position = new Vector3(4.0f, 3.0f, -4.0f);
                collision_flag = false;
                SpherePosition = new Vector3(4.0f, 0.0f, -4.0f);
                bonus1.transform.position = new Vector3(bonus1.transform.position.x, 0.6f, bonus1.transform.position.z);
                bonus2.transform.position = new Vector3(bonus2.transform.position.x, 0.6f, bonus2.transform.position.z);
                bonus3.transform.position = new Vector3(bonus3.transform.position.x, 0.6f, bonus3.transform.position.z);
                bonus4.transform.position = new Vector3(bonus4.transform.position.x, 0.6f, bonus4.transform.position.z);
                bonus5.transform.position = new Vector3(bonus5.transform.position.x, 0.6f, bonus5.transform.position.z);

            }
           
        }
        
    }

    void OnDirection(int direction, int done)
    {
        Debug.Log("CharacterMove : " + direction);

        switch(direction)
        {
            case 0: //위쪽 방향 -z방향
                move = new Vector3(0, 0, -1f) * 0.5f;
                break;
            case 1: //아래쪽 방향 z방향
                move = new Vector3(0, 0, 1f) * 0.5f;
                break;
            case 2: //오른쪽 방향 -x 방향
                move = new Vector3(-1f, 0, 0) * 0.5f;
                break;
            case 3: // 왼쪽 방향 x 방향
                move = new Vector3(1f, 0, 0) * 0.5f;
                break;
        }
        this.game_done = done;

        SubjectPosition = SpherePosition + move;

    }
}
