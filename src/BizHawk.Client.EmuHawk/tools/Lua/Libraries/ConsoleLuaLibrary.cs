﻿using System;
using System.Linq;
using System.Text;

using BizHawk.Client.Common;
using NLua;

namespace BizHawk.Client.EmuHawk
{
	public sealed class ConsoleLuaLibrary : LuaLibraryBase
	{
		public ToolManager Tools { get; set; }

		public ConsoleLuaLibrary(IPlatformLuaLibEnv luaLibsImpl, ApiContainer apiContainer, Action<string> logOutputCallback)
			: base(luaLibsImpl, apiContainer, logOutputCallback) {}

		public override string Name => "console";

		[LuaMethodExample("console.clear( );")]
		[LuaMethod("clear", "clears the output box of the Lua Console window")]
		public void Clear()
		{
			if (Tools.Has<LuaConsole>())
			{
				Tools.LuaConsole.ClearOutputWindow();
			}
		}

		[LuaMethodExample("local stconget = console.getluafunctionslist( );")]
		[LuaMethod("getluafunctionslist", "returns a list of implemented functions")]
		[return: LuaASCIIStringParam]
		public string GetLuaFunctionsList()
		{
			var list = new StringBuilder();
			foreach (var function in _luaLibsImpl.Docs)
			{
				list.AppendLine(function.Name);
			}

			return list.ToString();
		}

		[LuaMethodExample("console.log( \"New log.\" );")]
		[LuaMethod("log", "Outputs the given object to the output box on the Lua Console dialog. Note: Can accept a LuaTable")]
		public void Log([LuaArbitraryStringParam] params object[] outputs)
		{
			LogWithSeparator("\t", "\n", outputs);
		}

		[LuaMethodExample("console.writeline( \"New log line.\" );")]
		[LuaMethod("writeline", "Outputs the given object to the output box on the Lua Console dialog. Note: Can accept a LuaTable")]
		public void WriteLine([LuaArbitraryStringParam] params object[] outputs)
		{
			LogWithSeparator("\n", "\n", outputs);
		}

		[LuaMethodExample("console.write( \"New log message.\" );")]
		[LuaMethod("write", "Outputs the given object to the output box on the Lua Console dialog. Note: Can accept a LuaTable")]
		public void Write([LuaArbitraryStringParam] params object[] outputs)
		{
			LogWithSeparator("", "", outputs);
		}

		// Outputs the given object to the output box on the Lua Console dialog. Note: Can accept a LuaTable
		private void LogWithSeparator(string separator, string terminator, [LuaArbitraryStringParam] params object[] outputs)
		{
			static string SerializeTable([LuaArbitraryStringParam] LuaTable lti)
			{
				var keyObjs = lti.Keys;
				var valueObjs = lti.Values;
				if (keyObjs.Count != valueObjs.Count)
				{
					throw new ArgumentException(message: "each value must be paired with one key, they differ in number", paramName: nameof(lti));
				}

				var values = new object[keyObjs.Count];
				var kvpIndex = 0;
				foreach (var valueObj in valueObjs)
				{
					values[kvpIndex++] = valueObj;
				}

				return string.Concat(keyObjs.Cast<object>()
					.Select((kObj, i) => $"\"{(kObj is string s ? FixString(s) : kObj.ToString())}\": \"{(values[i] is string s1 ? FixString(s1) : values[i].ToString())}\"\n")
					.OrderBy(static s => s));
			}

			if (!Tools.Has<LuaConsole>())
			{
				return;
			}

			var sb = new StringBuilder();

			void SerializeAndWrite([LuaArbitraryStringParam] object output)
				=> sb.Append(output switch
				{
					null => "nil",
					LuaTable table => SerializeTable(table),
					string s => FixString(s),
					_ => output.ToString()
				});

			if (outputs == null)
			{
				sb.Append($"(no return){terminator}");
				return;
			}

			SerializeAndWrite(outputs[0]);
			for (int outIndex = 1, indexAfterLast = outputs.Length; outIndex != indexAfterLast; outIndex++)
			{
				sb.Append(separator);
				SerializeAndWrite(outputs[outIndex]);
			}

			if (!string.IsNullOrEmpty(terminator))
			{
				sb.Append(terminator);
			}

			Tools.LuaConsole.WriteToOutputWindow(sb.ToString());
		}
	}
}
