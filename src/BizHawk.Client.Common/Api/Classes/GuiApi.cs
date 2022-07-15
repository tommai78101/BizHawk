﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.IO;

using BizHawk.Emulation.Common;

namespace BizHawk.Client.Common
{
	public sealed class GuiApi : IGuiApi
	{
		[RequiredService]
		private IEmulator Emulator { get; set; }

		private readonly Action<string> LogCallback;

		private readonly DisplayManagerBase _displayManager;

		private readonly Dictionary<string, Image> _imageCache = new Dictionary<string, Image>();

		private readonly Bitmap _nullGraphicsBitmap = new Bitmap(1, 1);

		private readonly Dictionary<Color, Pen> _pens = new Dictionary<Color, Pen>();

		private readonly Dictionary<Color, SolidBrush> _solidBrushes = new Dictionary<Color, SolidBrush>();

		private ImageAttributes _attributes = new ImageAttributes();

		private CompositingMode _compositingMode = CompositingMode.SourceOver;

		private Color? _defaultBackground;

		private Color _defaultForeground = Color.White;

		private int _defaultPixelFont = 1; // = "gens"

		private Color? _defaultTextBackground = Color.FromArgb(128, 0, 0, 0);

		private IDisplaySurface _clientSurface;

		private IDisplaySurface _GUISurface;

		private (int Left, int Top, int Right, int Bottom) _padding = (0, 0, 0, 0);

		private DisplaySurfaceID? _usingSurfaceID = null;

		public bool EnableLuaAutolockHack = false;

		public bool HasGUISurface => _GUISurface != null;

		public GuiApi(Action<string> logCallback, DisplayManagerBase displayManager)
		{
			LogCallback = logCallback;
			_displayManager = displayManager;
		}

		private SolidBrush GetBrush(Color color) => _solidBrushes.TryGetValue(color, out var b) ? b : (_solidBrushes[color] = new SolidBrush(color));

		private Pen GetPen(Color color) => _pens.TryGetValue(color, out var p) ? p : (_pens[color] = new Pen(color));

		private Graphics GetGraphics(DisplaySurfaceID? surfaceID)
		{
			var g = GetRelevantSurface(surfaceID)?.GetGraphics() ?? Graphics.FromImage(_nullGraphicsBitmap);
			return g;
		}

		public void ToggleCompositingMode()
			=> _compositingMode = (CompositingMode) (1 - (int) _compositingMode); // enum has two members, 0 and 1

		public ImageAttributes GetAttributes() => _attributes;

		public void SetAttributes(ImageAttributes a) => _attributes = a;

		private IDisplaySurface GetRelevantSurface(DisplaySurfaceID? surfaceID)
		{
			var nnID = surfaceID ?? _usingSurfaceID ?? throw new Exception();
			void ThisIsTheLuaAutolockHack()
			{
				try
				{
					UnlockSurface(nnID);
					LockSurface(nnID);
				}
				catch (InvalidOperationException ex)
				{
					LogCallback(ex.ToString());
				}
			}
			switch (nnID)
			{
				case DisplaySurfaceID.EmuCore:
					if (_GUISurface == null && EnableLuaAutolockHack) ThisIsTheLuaAutolockHack();
					return _GUISurface;
				case DisplaySurfaceID.Client:
					if (_clientSurface == null && EnableLuaAutolockHack) ThisIsTheLuaAutolockHack();
					return _clientSurface;
				default:
					throw new Exception();
			}
		}

		private void LockSurface(DisplaySurfaceID surfaceID)
		{
			switch (surfaceID)
			{
				case DisplaySurfaceID.EmuCore:
					if (_GUISurface != null) throw new InvalidOperationException("attempt to lock surface without unlocking previous");
					_GUISurface = _displayManager.LockApiHawkSurface(surfaceID, clear: true);
					break;
				case DisplaySurfaceID.Client:
					if (_clientSurface != null) throw new InvalidOperationException("attempt to lock surface without unlocking previous");
					_clientSurface = _displayManager.LockApiHawkSurface(surfaceID, clear: true);
					break;
				default:
					throw new ArgumentException(message: "not a valid enum member", paramName: nameof(surfaceID));
			}
		}

		private void UnlockSurface(DisplaySurfaceID surfaceID)
		{
			switch (surfaceID)
			{
				case DisplaySurfaceID.EmuCore:
					if (_GUISurface != null) _displayManager.UnlockApiHawkSurface(_GUISurface);
					_GUISurface = null;
					break;
				case DisplaySurfaceID.Client:
					if (_clientSurface != null) _displayManager.UnlockApiHawkSurface(_clientSurface);
					_clientSurface = null;
					break;
				default:
					throw new ArgumentException(message: "not a valid enum member", paramName: nameof(surfaceID));
			}
		}

		public void WithSurface(DisplaySurfaceID surfaceID, Action drawingCallsFunc)
		{
			LockSurface(surfaceID);
			_usingSurfaceID = surfaceID;
			try
			{
				drawingCallsFunc();
			}
			finally
			{
				_usingSurfaceID = null;
				UnlockSurface(surfaceID);
			}
		}

		public void ThisIsTheLuaAutounlockHack()
		{
			UnlockSurface(DisplaySurfaceID.EmuCore);
			UnlockSurface(DisplaySurfaceID.Client);
		}

		public void DrawNew(string name, bool clear)
		{
			switch (name)
			{
				case null:
				case "emu":
					LogCallback("the `DrawNew(\"emu\")` function has been deprecated");
					return;
				case "native":
					throw new InvalidOperationException("the ability to draw in the margins with `DrawNew(\"native\")` has been removed");
				default:
					throw new InvalidOperationException("invalid surface name");
			}
		}

		public void DrawFinish() => LogCallback("the `DrawFinish()` function has been deprecated");

		public void SetPadding(int all) => _padding = (all, all, all, all);

		public void SetPadding(int x, int y) => _padding = (x / 2, y / 2, x / 2 + x & 1, y / 2 + y & 1);

		public void SetPadding(int l, int t, int r, int b) => _padding = (l, t, r, b);

		public (int Left, int Top, int Right, int Bottom) GetPadding() => _padding;

		public void AddMessage(string message, int? duration = null)
			=> _displayManager.OSD.AddMessage(message, duration);

		public void ClearGraphics(DisplaySurfaceID? surfaceID = null) => GetRelevantSurface(surfaceID).Clear();

		public void ClearText() => _displayManager.OSD.ClearGuiText();

		public void SetDefaultForegroundColor(Color color) => _defaultForeground = color;

		public void SetDefaultBackgroundColor(Color color) => _defaultBackground = color;

		public Color? GetDefaultTextBackground() => _defaultTextBackground;

		public void SetDefaultTextBackground(Color color) => _defaultTextBackground = color;

		public void SetDefaultPixelFont(string fontfamily)
		{
			switch (fontfamily)
			{
				case "fceux":
				case "0":
					_defaultPixelFont = 0;
					break;
				case "gens":
				case "1":
					_defaultPixelFont = 1;
					break;
				default:
					LogCallback($"Unable to find font family: {fontfamily}");
					return;
			}
		}

		public void DrawBezier(Point p1, Point p2, Point p3, Point p4, Color? color = null, DisplaySurfaceID? surfaceID = null)
		{
			try
			{
				using var g = GetGraphics(surfaceID);
				g.CompositingMode = _compositingMode;
				g.DrawBezier(GetPen(color ?? _defaultForeground), p1, p2, p3, p4);
			}
			catch (Exception)
			{
				// ignored
			}
		}

		public void DrawBeziers(Point[] points, Color? color = null, DisplaySurfaceID? surfaceID = null)
		{
			try
			{
				using var g = GetGraphics(surfaceID);
				g.CompositingMode = _compositingMode;
				g.DrawBeziers(GetPen(color ?? _defaultForeground), points);
			}
			catch (Exception)
			{
				// ignored
			}
		}

		public void DrawBox(int x, int y, int x2, int y2, Color? line = null, Color? background = null, DisplaySurfaceID? surfaceID = null)
		{
			try
			{
				float w;
				if (x < x2)
				{
					w = x2 - x;
				}
				else
				{
					x2 = x - x2;
					x -= x2;
					w = Math.Max(x2, 0.1f);
				}
				float h;
				if (y < y2)
				{
					h = y2 - y;
				}
				else
				{
					y2 = y - y2;
					y -= y2;
					h = Math.Max(y2, 0.1f);
				}
				using var g = GetGraphics(surfaceID);
				g.CompositingMode = _compositingMode;
				g.DrawRectangle(GetPen(line ?? _defaultForeground), x, y, w, h);
				var bg = background ?? _defaultBackground;
				if (bg != null) g.FillRectangle(GetBrush(bg.Value), x + 1, y + 1, Math.Max(w - 1, 0), Math.Max(h - 1, 0));
			}
			catch (Exception)
			{
				// need to stop the script from here
			}
		}

		public void DrawEllipse(int x, int y, int width, int height, Color? line = null, Color? background = null, DisplaySurfaceID? surfaceID = null)
		{
			try
			{
				using var g = GetGraphics(surfaceID);
				var bg = background ?? _defaultBackground;
				if (bg != null) g.FillEllipse(GetBrush(bg.Value), x, y, width, height);
				g.CompositingMode = _compositingMode;
				g.DrawEllipse(GetPen(line ?? _defaultForeground), x, y, width, height);
			}
			catch (Exception)
			{
				// need to stop the script from here
			}
		}

		public void DrawIcon(string path, int x, int y, int? width = null, int? height = null, DisplaySurfaceID? surfaceID = null)
		{
			try
			{
				if (!File.Exists(path))
				{
					AddMessage($"File not found: {path}");
					return;
				}
				using var g = GetGraphics(surfaceID);
				g.CompositingMode = _compositingMode;
				g.DrawIcon(
					width != null && height != null
						? new Icon(path, width.Value, height.Value)
						: new Icon(path),
					x,
					y
				);
			}
			catch (Exception)
			{
				// ignored
			}
		}

		public void DrawImage(Image img, int x, int y, int? width = null, int? height = null, bool cache = true, DisplaySurfaceID? surfaceID = null)
		{
			using var g = GetGraphics(surfaceID);
			g.CompositingMode = _compositingMode;
			g.DrawImage(
				img,
				new Rectangle(x, y, width ?? img.Width, height ?? img.Height),
				0,
				0,
				img.Width,
				img.Height,
				GraphicsUnit.Pixel,
				_attributes
			);
		}
		public void DrawImage(string path, int x, int y, int? width = null, int? height = null, bool cache = true, DisplaySurfaceID? surfaceID = null)
		{
			if (!File.Exists(path))
			{
				LogCallback($"File not found: {path}");
				return;
			}
			using var g = GetGraphics(surfaceID);
			var isCached = _imageCache.ContainsKey(path);
			var img = isCached ? _imageCache[path] : Image.FromFile(path);
			if (!isCached && cache) _imageCache[path] = img;
			g.CompositingMode = _compositingMode;
			g.DrawImage(
				img,
				new Rectangle(x, y, width ?? img.Width, height ?? img.Height),
				0,
				0,
				img.Width,
				img.Height,
				GraphicsUnit.Pixel,
				_attributes
			);
		}

		public void ClearImageCache()
		{
			foreach (var image in _imageCache) image.Value.Dispose();
			_imageCache.Clear();
		}

		public void DrawImageRegion(Image img, int source_x, int source_y, int source_width, int source_height, int dest_x, int dest_y, int? dest_width = null, int? dest_height = null, DisplaySurfaceID? surfaceID = null)
		{
			using var g = GetGraphics(surfaceID);
			g.CompositingMode = _compositingMode;
			g.DrawImage(
				img,
				new Rectangle(dest_x, dest_y, dest_width ?? source_width, dest_height ?? source_height),
				source_x,
				source_y,
				source_width,
				source_height,
				GraphicsUnit.Pixel,
				_attributes
			);
		}

		public void DrawImageRegion(string path, int source_x, int source_y, int source_width, int source_height, int dest_x, int dest_y, int? dest_width = null, int? dest_height = null, DisplaySurfaceID? surfaceID = null)
		{
			if (!File.Exists(path))
			{
				LogCallback($"File not found: {path}");
				return;
			}
			using var g = GetGraphics(surfaceID);
			g.CompositingMode = _compositingMode;
			g.DrawImage(
				_imageCache.TryGetValue(path, out var img) ? img : (_imageCache[path] = Image.FromFile(path)),
				new Rectangle(dest_x, dest_y, dest_width ?? source_width, dest_height ?? source_height),
				source_x,
				source_y,
				source_width,
				source_height,
				GraphicsUnit.Pixel,
				_attributes
			);
		}

		public void DrawLine(int x1, int y1, int x2, int y2, Color? color = null, DisplaySurfaceID? surfaceID = null)
		{
			using var g = GetGraphics(surfaceID);
			g.CompositingMode = _compositingMode;
			g.DrawLine(GetPen(color ?? _defaultForeground), x1, y1, x2, y2);
		}

		public void DrawAxis(int x, int y, int size, Color? color = null, DisplaySurfaceID? surfaceID = null)
		{
			DrawLine(x + size, y, x - size, y, color ?? _defaultForeground, surfaceID: surfaceID);
			DrawLine(x, y + size, x, y - size, color ?? _defaultForeground, surfaceID: surfaceID);
		}

		public void DrawPie(int x, int y, int width, int height, int startangle, int sweepangle, Color? line = null, Color? background = null, DisplaySurfaceID? surfaceID = null)
		{
			using var g = GetGraphics(surfaceID);
			g.CompositingMode = _compositingMode;
			var bg = background ?? _defaultBackground;
			if (bg != null) g.FillPie(GetBrush(bg.Value), x, y, width, height, startangle, sweepangle);
			g.DrawPie(GetPen(line ?? _defaultForeground), x + 1, y + 1, width - 1, height - 1, startangle, sweepangle);
		}

		public void DrawPixel(int x, int y, Color? color = null, DisplaySurfaceID? surfaceID = null)
		{
			try
			{
				using var g = GetGraphics(surfaceID);
				g.DrawLine(GetPen(color ?? _defaultForeground), x, y, x + 0.1F, y);
			}
			catch (Exception)
			{
				// ignored
			}
		}

		public void DrawPolygon(Point[] points, Color? line = null, Color? background = null, DisplaySurfaceID? surfaceID = null)
		{
			try
			{
				using var g = GetGraphics(surfaceID);
				g.DrawPolygon(GetPen(line ?? _defaultForeground), points);
				var bg = background ?? _defaultBackground;
				if (bg != null) g.FillPolygon(GetBrush(bg.Value), points);
			}
			catch (Exception)
			{
				// ignored
			}
		}

		public void DrawRectangle(int x, int y, int width, int height, Color? line = null, Color? background = null, DisplaySurfaceID? surfaceID = null)
		{
			using var g = GetGraphics(surfaceID);
			var w = Math.Max(width, 0.1F);
			var h = Math.Max(height, 0.1F);
			g.DrawRectangle(GetPen(line ?? _defaultForeground), x, y, w, h);
			var bg = background ?? _defaultBackground;
			if (bg != null) g.FillRectangle(GetBrush(bg.Value), x + 1, y + 1, Math.Max(w - 1, 0), Math.Max(h - 1, 0));
		}

		public void DrawString(int x, int y, string message, Color? forecolor = null, Color? backcolor = null, int? fontsize = null, string fontfamily = null, string fontstyle = null, string horizalign = null, string vertalign = null, DisplaySurfaceID? surfaceID = null)
		{
			try
			{
				var family = fontfamily != null ? new FontFamily(fontfamily) : FontFamily.GenericMonospace;

				var fstyle = fontstyle?.ToLower() switch
				{
					"bold" => FontStyle.Bold,
					"italic" => FontStyle.Italic,
					"strikethrough" => FontStyle.Strikeout,
					"underline" => FontStyle.Underline,
					_ => FontStyle.Regular
				};

				using var g = GetGraphics(surfaceID);

				// The text isn't written out using GenericTypographic, so measuring it using GenericTypographic seemed to make it worse.
				// And writing it out with GenericTypographic just made it uglier. :p
				var font = new Font(family, fontsize ?? 12, fstyle, GraphicsUnit.Pixel);
				var sizeOfText = g.MeasureString(message, font, 0, new StringFormat(StringFormat.GenericDefault)).ToSize();

				switch (horizalign?.ToLower())
				{
					default:
					case "left":
						break;
					case "center":
					case "middle":
						x -= sizeOfText.Width / 2;
						break;
					case "right":
						x -= sizeOfText.Width;
						break;
				}

				switch (vertalign?.ToLower())
				{
					default:
					case "top":
						break;
					case "center":
					case "middle":
						y -= sizeOfText.Height / 2;
						break;
					case "bottom":
						y -= sizeOfText.Height;
						break;
				}

				var bg = backcolor ?? _defaultBackground;
				if (bg != null)
				{
					var brush = GetBrush(bg.Value);
					for (var xd = -1; xd <= 1; xd++) for (var yd = -1; yd <= 1; yd++)
					{
						g.DrawString(message, font, brush, x + xd, y + yd);
					}
				}
				g.TextRenderingHint = TextRenderingHint.SingleBitPerPixelGridFit;
				g.DrawString(message, font, GetBrush(forecolor ?? _defaultForeground), x, y);
			}
			catch (Exception)
			{
				// ignored
			}
		}

		public void DrawText(int x, int y, string message, Color? forecolor = null, Color? backcolor = null, string fontfamily = null, DisplaySurfaceID? surfaceID = null)
		{
			try
			{
				int index;
				switch (fontfamily)
				{
					case "fceux":
					case "0":
						index = 0;
						break;
					case "gens":
					case "1":
						index = 1;
						break;
					default:
						if (!string.IsNullOrEmpty(fontfamily)) // not a typo
						{
							LogCallback($"Unable to find font family: {fontfamily}");
							return;
						}
						index = _defaultPixelFont;
						break;
				}
				using var g = GetGraphics(surfaceID);
				var font = new Font(_displayManager.CustomFonts.Families[index], 8, FontStyle.Regular, GraphicsUnit.Pixel);
				var sizeOfText = g.MeasureString(
					message,
					font,
					0,
					new StringFormat(StringFormat.GenericTypographic) { FormatFlags = StringFormatFlags.MeasureTrailingSpaces }
				).ToSize();
				if (backcolor.HasValue) g.FillRectangle(GetBrush(backcolor.Value), new Rectangle(new Point(x, y), sizeOfText + new Size(1, 0)));
				g.TextRenderingHint = TextRenderingHint.SingleBitPerPixelGridFit;
				g.DrawString(message, font, GetBrush(forecolor ?? _defaultForeground), x, y);
			}
			catch (Exception)
			{
				// ignored
			}
		}

		public void Text(int x, int y, string message, Color? forecolor = null, string anchor = null)
		{
			int a = default;
			if (!string.IsNullOrEmpty(anchor))
			{
				a = anchor switch
				{
					"0" => 0,
					"topleft" => 0,
					"1" => 1,
					"topright" => 1,
					"2" => 2,
					"bottomleft" => 2,
					"3" => 3,
					"bottomright" => 3,
					_ => default
				};
			}
			else
			{
				var (ox, oy) = Emulator.ScreenLogicalOffsets();
				x -= ox;
				y -= oy;
			}

			var pos = new MessagePosition{ X = x, Y = y, Anchor = (MessagePosition.AnchorType)a };
			_displayManager.OSD.AddGuiText(message,  pos, Color.Black, forecolor ?? Color.White);
		}

		public void Dispose()
		{
			UnlockSurface(DisplaySurfaceID.EmuCore);
			UnlockSurface(DisplaySurfaceID.Client);
			foreach (var brush in _solidBrushes.Values) brush.Dispose();
			foreach (var brush in _pens.Values) brush.Dispose();
		}
	}
}