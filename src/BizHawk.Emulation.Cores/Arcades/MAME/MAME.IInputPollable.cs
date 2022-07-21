﻿using System;
using System.Collections.Generic;

using BizHawk.Common.StringExtensions;
using BizHawk.Emulation.Common;

namespace BizHawk.Emulation.Cores.Arcades.MAME
{
	public partial class MAME : IInputPollable
	{
		public int LagCount { get; set; } = 0;
		public bool IsLagFrame { get; set; } = false;

		[FeatureNotImplemented]
		public IInputCallbackSystem InputCallbacks => throw new NotImplementedException();

		public static ControllerDefinition MAMEController = new("MAME Controller");

		private IController _controller = NullController.Instance;
		private readonly SortedDictionary<string, string> _fieldsPorts = new SortedDictionary<string, string>();
		private SortedDictionary<string, string> _romHashes = new SortedDictionary<string, string>();

		private void GetInputFields()
		{
			string inputFields = MameGetString(MAMELuaCommand.GetInputFields);
			string[] portFields = inputFields.Split(';');
			MAMEController = new("MAME Controller");
			MAMEController.BoolButtons.Clear();
			_fieldsPorts.Clear();

			foreach (string portField in portFields)
			{
				if (portField != string.Empty)
				{
					var tag = portField.SubstringBefore(',');
					var field = portField.SubstringAfterLast(',');
					_fieldsPorts.Add(field, tag);
					MAMEController.BoolButtons.Add(field);
				}
			}

			MAMEController.MakeImmutable();
		}

		private void SendInput()
		{
			foreach (var fieldPort in _fieldsPorts)
			{
				LibMAME.mame_lua_execute(
					"manager.machine.ioport" +
					$".ports  [\"{ fieldPort.Value }\"]" +
					$".fields [\"{ fieldPort.Key   }\"]" +
					$":set_value({ (_controller.IsPressed(fieldPort.Key) ? 1 : 0) })");
			}
		}
	}
}