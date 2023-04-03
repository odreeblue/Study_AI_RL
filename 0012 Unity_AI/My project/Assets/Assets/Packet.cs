using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using UnityEngine;

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
struct TouchPacket
{
    [MarshalAs(UnmanagedType.I4)]
    public int typeOfService;
    [MarshalAs(UnmanagedType.I4)]
    public int displayId;
    [MarshalAs(UnmanagedType.I4)]
    public int payloadLength;
    [MarshalAs(UnmanagedType.I4)]
    public int x;
    [MarshalAs(UnmanagedType.I4)]
    public int y;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
struct DirectionPacket
{
    [MarshalAs(UnmanagedType.I4)]
    public int typeOfService;
    [MarshalAs(UnmanagedType.I4)]
    public int displayId;
    [MarshalAs(UnmanagedType.I4)]
    public int payloadLength;
    [MarshalAs(UnmanagedType.I4)]
    public int direction;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
struct GazePacket
{
    [MarshalAs(UnmanagedType.I4)]
    public int typeOfService;
    [MarshalAs(UnmanagedType.I4)]
    public int displayId;
    [MarshalAs(UnmanagedType.I4)]
    public int payloadLength;
    [MarshalAs(UnmanagedType.I4)]
    public int x;
    [MarshalAs(UnmanagedType.I4)]
    public int y;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
struct VoicePacket
{
    [MarshalAs(UnmanagedType.I4)]
    public int typeOfService;
    [MarshalAs(UnmanagedType.I4)]
    public int displayId;
    [MarshalAs(UnmanagedType.I4)]
    public int payloadLength;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 1012)]
    public string message;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
struct HandSkeletonPacket
{
    [MarshalAs(UnmanagedType.I4)]
    public int typeOfService;
    [MarshalAs(UnmanagedType.I4)]
    public int displayId;
    [MarshalAs(UnmanagedType.I4)]
    public int payloadLength;
    [MarshalAs(UnmanagedType.I4)]
    public int x;
    [MarshalAs(UnmanagedType.I4)]
    public int y;
}


public class PacketManager
{
    // Packet to send
    private TouchPacket touchPacket = new TouchPacket();
    private DirectionPacket directionPacket = new DirectionPacket();
    private GazePacket gazePacket = new GazePacket();
    private VoicePacket voicePacket = new VoicePacket();
    private HandSkeletonPacket handSkeletonPacket = new HandSkeletonPacket();

    // Display Id
    private int displayId;
    
    public PacketManager(int Id)
    {
        displayId = Id;
    }

    public byte[] GetTouchPacket(int x, int y)
    {
        touchPacket.typeOfService = 0;
        touchPacket.displayId = displayId;
        touchPacket.payloadLength = 8;
        touchPacket.x = x;
        touchPacket.y = y;

        return Serialize<TouchPacket>(touchPacket);
    }

    public byte[] GetDirectionPacket(int direction)
    {
        directionPacket.typeOfService = 1;
        directionPacket.displayId = displayId;
        directionPacket.payloadLength = 4;
        directionPacket.direction = direction;

        return Serialize<DirectionPacket>(directionPacket);
    }

    public byte[] GetGazePacket(int x, int y)
    {
        gazePacket.typeOfService = 2;
        gazePacket.displayId = displayId;
        gazePacket.payloadLength = 8;
        gazePacket.x = x;
        gazePacket.y = y;

        return Serialize<GazePacket>(gazePacket);
    }

    public byte[] GetVoicePacket(string message)
    {
        if(message.Length >= 1012)
        {
            Debug.Log("Exceed to expected length (>= 1012");
            message = "Error";
        }
        voicePacket.typeOfService = 3;
        voicePacket.displayId = displayId;
        voicePacket.payloadLength = 1012;
        voicePacket.message = message;

        return Serialize<VoicePacket>(voicePacket);
    }

    public byte[] GetHandSkeletonPacket(int x, int y)
    {
        handSkeletonPacket.typeOfService = 5;
        handSkeletonPacket.displayId = displayId;
        handSkeletonPacket.payloadLength = 8;
        handSkeletonPacket.x = x;
        handSkeletonPacket.y = y;

        return Serialize<HandSkeletonPacket>(handSkeletonPacket);
    }


    // Calling this method will return a byte array with the contents
    // of the struct ready to be sent via the tcp socket.
    private byte[] Serialize<T>(T packet)
    {
        // allocate a byte array for the struct data
        var buffer = new byte[Marshal.SizeOf(typeof(T))];

        // Allocate a GCHandle and get the array pointer
        var gch = GCHandle.Alloc(buffer, GCHandleType.Pinned);
        var pBuffer = gch.AddrOfPinnedObject();

        // copy data from struct to array and unpin the gc pointer
        Marshal.StructureToPtr(packet, pBuffer, false);
        gch.Free();

        return buffer;
    }
} 

