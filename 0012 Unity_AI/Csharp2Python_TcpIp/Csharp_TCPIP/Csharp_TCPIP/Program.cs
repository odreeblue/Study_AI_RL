using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections;
using System.Runtime.InteropServices;

namespace server
{
    class Program
    {
        public const int port = 3000;

        //public static ArrayList clientList = new ArrayList();
        static void Main(string[] args)
        {
            new Program();
        }

        public Program()
        {
            TcpListener tcpListener = new TcpListener(IPAddress.Parse("192.168.200.108"), port);
            tcpListener.Start();
            while (true)
            {
                //Socket client = tcpListener.AcceptSocket();
                TcpClient client = tcpListener.AcceptTcpClient();
                //IPEndPoint ip = (IPEndPoint)client.RemoteEndPoint;
                //IPEndPoint ip = (IPEndPoint)client.;
                Console.WriteLine("주소 {0}에서 접속", client.ToString());
                ClientListener clientThread = new ClientListener(client);
                //clientList.Add(clientThread);
            }
        }
    }

    class ClientListener
    {
        //String id;
        TcpClient client;
        NetworkStream stream;

        public ClientListener(TcpClient client)
        {
            this.client = client;
            //NetworkStream networkStream = new NetworkStream(client);
            //NetworkStream networkStream = client.
            this.stream = client.GetStream();

            //this.reader = new System.IO.StreamReader(networkStream);
            //this.reader = new System.IO.(networkStream);
            //this.writer = new System.IO.StreamWriter(networkStream);
            (new Thread(new ThreadStart(Run))).Start();
        }

        private void Run()
        {
            DataPacket packet = new DataPacket();
            //packet.Name = "ChanYeong";
            //packet.Subject = "Math";
            //packet.Grade = 4;
            //packet.Memo = "GoodLuck";


            while (true)
            {
                Console.WriteLine("Name : ");
                string Name = Console.ReadLine();
                packet.Name = Name;
                Console.WriteLine("Subject : ");
                string Subject= Console.ReadLine();
                packet.Subject = Subject;
                Console.WriteLine("Grade : ");
                string Grade = Console.ReadLine();
                Int32 Grade2 = Convert.ToInt32(Grade);
                packet.Grade = Grade2;
                Console.WriteLine("Memo : ");
                string Memo = Console.ReadLine();
                packet.Memo = Memo;



                byte[] buffer = new byte[Marshal.SizeOf(packet)];
                unsafe
                {
                    fixed (byte* fixed_buffer = buffer)
                    {
                        Marshal.StructureToPtr(packet, (IntPtr)fixed_buffer, false);
                    }
                }
                //byte[] buffer2 = new byte[100];
                //stream.Read(buffer2, 0, );
                //String line = "";
                //while (line.Equals(""))
                //{
                //    line = stream.Read(buffer);
                //    if (line == null)
                //    {
                //        //Program.clientList.Remove(this);
                //        Console.WriteLine(this.id + "접속 종료");
                //        return;
                //    }
                //}
                //Console.WriteLine(client.ToString() + "-send/" + line);

                stream.Write(buffer, 0, Marshal.SizeOf(packet));
                Console.WriteLine("packet의 사이즈는 :  "+Marshal.SizeOf(packet));
                //writer.WriteLine("ohhhhyeahhhhh");
                stream.Flush();

                
                byte[] buffer2 = new byte[Marshal.SizeOf(packet)];
                stream.Read(buffer2, 0, Marshal.SizeOf(packet));
                unsafe
                {
                    fixed (byte* fixed_buffer = buffer2)
                    {
                        Marshal.PtrToStructure((IntPtr)fixed_buffer, packet);
                    }
                }
                Name = packet.Name;
                Subject = packet.Subject;
                int Grade3 = packet.Grade;
                Memo = packet.Memo;

                Console.WriteLine("이름 : {0}", Name);
                Console.WriteLine("과목 : {0}", Subject);
                Console.WriteLine("점수 : {0}", Grade3);
                Console.WriteLine("메모 : {0}", Memo);
                Console.WriteLine("");
                Console.WriteLine("===========================");
                Console.WriteLine("");


                //String[] command = line.Split("/");
                //if (command[0].Equals("addMsg"))
                //{
                //    foreach (object clientThread in Program.clientList)
                //    {
                //        ClientListener client = (ClientListener)clientThread;
                //        Console.WriteLine(client.id);
                //        if (!client.Equals(this))
                //        {
                //            NetworkStream networkStream = new NetworkStream(client.client);
                //            System.IO.StreamWriter writer = new System.IO.StreamWriter(networkStream);
                //            writer.WriteLine("addMsg/" + this.id + "/" + command[1]);
                //            writer.Flush();
                //        }
                //    }
                //}
                //else if (command[0].Equals("id"))
                //{
                //    this.id = command[1];
                //}
                //stream.Close();
                //client.Close();
            }
        }
        //public override bool Equals(Object obj)
        //{
        //    //Check for null and compare run-time types.
        //    if ((obj == null) || !this.GetType().Equals(obj.GetType()))
        //    {
        //        return false;
        //    }
        //    else
        //    {
        //        ClientListener other = (ClientListener)obj;
        //        return (this.id == other.id);
        //    }
        //}
        //public override int GetHashCode() { return 0; } // 나중에 알아보기
    }
}
[StructLayout(LayoutKind.Sequential)]
class DataPacket
{
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 20)]
    public string Name;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 20)]
    public string Subject;
    [MarshalAs(UnmanagedType.I4)]
    public int Grade;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 100)]
    public string Memo;
}
