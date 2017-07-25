#if UNITY_EDITOR
using UnityEditor;
using System.Diagnostics;
using System.IO.Compression;
using System.IO;
using System;

public class BuildAndTransferConfiguration
{
    [MenuItem("Build Tools/Build and Transfer Configuration")]
    public static void BuildGame()
    {
        // Get filename.
        string path = EditorUtility.SaveFolderPanel("Choose Location of Built Game", "", "");
        
        string[] levels = new string[] { "Assets/Scenes/Menu.unity", "Assets/Scenes/MainRoom.unity" };

        // Build player.
        BuildPipeline.BuildPlayer(levels, path + "/VirtualMorrisWaterMaze.exe", BuildTarget.StandaloneWindows, BuildOptions.None);

        // Copy a file from the project folder to the build folder, alongside the built game.
        FileUtil.CopyFileOrDirectory("Assets/configuration.ini", path + "/VirtualMorrisWaterMaze_Data/configuration.ini");

        //string[] split_dirs = path.Split(Path.DirectorySeparatorChar);
        //string outputPath = "/../" + split_dirs[split_dirs.Length - 1] + ".zip";
        //Console.WriteLine(outputPath);
        //CompressDirectory(path, outputPath, (fileName) => { Console.WriteLine("Compressing {0}...", fileName); });

        //File.Delete(path + "/player_win_x86.pdb");
    }

    delegate void ProgressDelegate(string sMessage);

    static void CompressFile(string sDir, string sRelativePath, GZipStream zipStream)
    {
        //Compress file name
        char[] chars = sRelativePath.ToCharArray();
        zipStream.Write(BitConverter.GetBytes(chars.Length), 0, sizeof(int));
        foreach (char c in chars)
            zipStream.Write(BitConverter.GetBytes(c), 0, sizeof(char));

        //Compress file content
        byte[] bytes = File.ReadAllBytes(Path.Combine(sDir, sRelativePath));
        zipStream.Write(BitConverter.GetBytes(bytes.Length), 0, sizeof(int));
        zipStream.Write(bytes, 0, bytes.Length);
    }

    static void CompressDirectory(string sInDir, string sOutFile, ProgressDelegate progress)
    {
        string[] sFiles = Directory.GetFiles(sInDir, "*.*", SearchOption.AllDirectories);
        int iDirLen = sInDir[sInDir.Length - 1] == Path.DirectorySeparatorChar ? sInDir.Length : sInDir.Length + 1;

        using (FileStream outFile = new FileStream(sOutFile, FileMode.Create, FileAccess.Write, FileShare.None))
        using (GZipStream str = new GZipStream(outFile, CompressionMode.Compress))
            foreach (string sFilePath in sFiles)
            {
                string sRelativePath = sFilePath.Substring(iDirLen);
                if (progress != null)
                    progress(sRelativePath);
                CompressFile(sInDir, sRelativePath, str);
            }
    }
}
#endif