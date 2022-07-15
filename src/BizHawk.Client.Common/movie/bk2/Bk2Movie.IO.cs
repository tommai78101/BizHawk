﻿using System;
using System.IO;

using BizHawk.Common;
using BizHawk.Common.IOExtensions;
using BizHawk.Emulation.Common;

namespace BizHawk.Client.Common
{
	public partial class Bk2Movie
	{
		public void Save()
		{
			Write(Filename);
		}

		public void SaveBackup()
		{
			if (string.IsNullOrWhiteSpace(Filename))
			{
				return;
			}

			var backupName = Filename;
			backupName = backupName.Insert(Filename.LastIndexOf("."), $".{DateTime.Now:yyyy-MM-dd HH.mm.ss}");
			backupName = Path.Combine(Session.BackupDirectory, Path.GetFileName(backupName));

			Write(backupName, isBackup: true);
		}

		public virtual bool Load(bool preload)
		{
			var file = new FileInfo(Filename);
			if (!file.Exists)
			{
				return false;
			}

			using var bl = ZipStateLoader.LoadAndDetect(Filename, true);
			if (bl == null)
			{
				return false;
			}

			ClearBeforeLoad();
			LoadFields(bl, preload);

			Changes = false;
			return true;
		}

		public bool PreLoadHeaderAndLength() => Load(true);

		protected virtual void Write(string fn, bool isBackup = false)
		{
			SetCycleValues();
			// EmulatorVersion used to store the unchanging original emulator version.
			if (!Header.ContainsKey(HeaderKeys.OriginalEmulatorVersion))
			{
				Header[HeaderKeys.OriginalEmulatorVersion] = Header[HeaderKeys.EmulatorVersion];
			}
			Header[HeaderKeys.EmulatorVersion] = VersionInfo.GetEmuVersion();
			CreateDirectoryIfNotExists(fn);

			using var bs = new ZipStateSaver(fn, Session.Settings.MovieCompressionLevel);
			AddLumps(bs, isBackup);

			if (!isBackup)
			{
				Changes = false;
			}
		}

		private void SetCycleValues()
		{
			// The saved cycle value will only be valid if the end of the movie has been emulated.
			if (this.IsAtEnd())
			{
				if (Emulator is ICycleTiming cycleCore)
				{
					Header[HeaderKeys.CycleCount] = cycleCore.CycleCount.ToString();
					Header[HeaderKeys.ClockRate] = cycleCore.ClockRate.ToString();
				}
			}
			else
			{
				Header.Remove(HeaderKeys.CycleCount);
				Header.Remove(HeaderKeys.ClockRate);
			}
		}

		private static void CreateDirectoryIfNotExists(string fn)
		{
			var file = new FileInfo(fn);
			if (file.Directory != null && !file.Directory.Exists)
			{
				Directory.CreateDirectory(file.Directory.ToString());
			}
		}

		protected virtual void AddLumps(ZipStateSaver bs, bool isBackup = false)
		{
			AddBk2Lumps(bs);
		}

		protected void AddBk2Lumps(ZipStateSaver bs)
		{
			bs.PutLump(BinaryStateLump.Movieheader, tw => tw.WriteLine(Header.ToString()));
			bs.PutLump(BinaryStateLump.Comments, tw => tw.WriteLine(CommentsString()));
			bs.PutLump(BinaryStateLump.Subtitles, tw => tw.WriteLine(Subtitles.ToString()));
			bs.PutLump(BinaryStateLump.SyncSettings, tw => tw.WriteLine(SyncSettingsJson));
			bs.PutLump(BinaryStateLump.Input, WriteInputLog);

			if (StartsFromSavestate)
			{
				if (TextSavestate != null)
				{
					bs.PutLump(BinaryStateLump.CorestateText, (TextWriter tw) => tw.Write(TextSavestate));
				}
				else
				{
					bs.PutLump(BinaryStateLump.Corestate, (BinaryWriter bw) => bw.Write(BinarySavestate));
				}

				if (SavestateFramebuffer != null)
				{
					bs.PutLump(BinaryStateLump.Framebuffer, (BinaryWriter bw) => bw.Write(SavestateFramebuffer));
				}
			}
			else if (StartsFromSaveRam)
			{
				bs.PutLump(BinaryStateLump.MovieSaveRam, (BinaryWriter bw) => bw.Write(SaveRam));
			}
		}

		protected virtual void ClearBeforeLoad()
		{
			ClearBk2Fields();
		}

		protected void ClearBk2Fields()
		{
			Header.Clear();
			Log.Clear();
			Subtitles.Clear();
			Comments.Clear();
			_syncSettingsJson = "";
			TextSavestate = null;
			BinarySavestate = null;
		}

		protected virtual void LoadFields(ZipStateLoader bl, bool preload)
		{
			LoadBk2Fields(bl, preload);
		}

		protected void LoadBk2Fields(ZipStateLoader bl, bool preload)
		{
			bl.GetLump(BinaryStateLump.Movieheader, abort: true, tr =>
			{
				string line;
				while ((line = tr.ReadLine()) != null)
				{
					if (!string.IsNullOrWhiteSpace(line))
					{
						var pair = line.Split(new[] { ' ' }, 2, StringSplitOptions.RemoveEmptyEntries);

						if (pair.Length > 1)
						{
							if (!Header.ContainsKey(pair[0]))
							{
								Header.Add(pair[0], pair[1]);
							}
						}
					}
				}
			});

			bl.GetLump(BinaryStateLump.Input, abort: true, tr =>
			{
				IsCountingRerecords = false;
				ExtractInputLog(tr, out _);
				IsCountingRerecords = true;
			});

			if (preload)
			{
				return;
			}

			bl.GetLump(BinaryStateLump.Comments, abort: false, tr =>
			{
				string line;
				while ((line = tr.ReadLine()) != null)
				{
					if (!string.IsNullOrWhiteSpace(line))
					{
						Comments.Add(line);
					}
				}
			});

			bl.GetLump(BinaryStateLump.Subtitles, abort: false, tr =>
			{
				string line;
				while ((line = tr.ReadLine()) != null)
				{
					if (!string.IsNullOrWhiteSpace(line))
					{
						Subtitles.AddFromString(line);
					}
				}

				Subtitles.Sort();
			});

			bl.GetLump(BinaryStateLump.SyncSettings, abort: false, tr =>
			{
				string line;
				while ((line = tr.ReadLine()) != null)
				{
					if (!string.IsNullOrWhiteSpace(line))
					{
						_syncSettingsJson = line;
					}
				}
			});

			if (StartsFromSavestate)
			{
				bl.GetCoreState(
					(br, length) => BinarySavestate = br.ReadBytes((int) length),
					tr => TextSavestate = tr.ReadToEnd());
				bl.GetLump(BinaryStateLump.Framebuffer, false,
					(br, length) =>
					{
						SavestateFramebuffer = new int[length / sizeof(int)];
						for (int i = 0; i < SavestateFramebuffer.Length; i++)
						{
							SavestateFramebuffer[i] = br.ReadInt32();
						}
					});
			}
			else if (StartsFromSaveRam)
			{
				bl.GetLump(BinaryStateLump.MovieSaveRam, false,
					(br, length) => SaveRam = br.ReadBytes((int) length));
			}
		}
	}
}
