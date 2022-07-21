using System;
using System.Linq;

namespace BizHawk.Common.StringExtensions
{
	public static class StringExtensions
	{
		public static string CharCodepointsToString(byte[] array)
		{
			var a = new char[array.Length];
			for (var i = 0; i < array.Length; i++) a[i] = char.ConvertFromUtf32(array[i])[0];
			return new(a);
		}

		public static bool Contains(this string haystack, string needle, StringComparison comparisonType)
			=> haystack.IndexOf(needle, comparisonType) != -1;

		/// <returns>
		/// <see langword="true"/> if <paramref name="str"/> appears in <paramref name="options"/> (case-insensitive)
		/// </returns>
		public static bool In(this string str, params string[] options) =>
			options.Any(opt => string.Equals(opt, str, StringComparison.InvariantCultureIgnoreCase));

		/// <returns>
		/// <paramref name="str"/> with the first char removed, or
		/// the original <paramref name="str"/> if the first char of <paramref name="str"/> is not <paramref name="prefix"/>
		/// </returns>
		public static string RemovePrefix(this string str, char prefix) => str.RemovePrefix(prefix, notFoundValue: str);

		/// <returns>
		/// <paramref name="str"/> with the first char removed, or
		/// <paramref name="notFoundValue"/> if the first char of <paramref name="str"/> is not <paramref name="prefix"/>
		/// </returns>
		public static string RemovePrefix(this string str, char prefix, string notFoundValue) => str.Length != 0 && str[0] == prefix ? str.Substring(1, str.Length - 1) : notFoundValue;

		/// <returns>
		/// <paramref name="str"/> with the leading substring <paramref name="prefix"/> removed, or
		/// the original <paramref name="str"/> if <paramref name="str"/> does not start with <paramref name="prefix"/>
		/// </returns>
		public static string RemovePrefix(this string str, string prefix) => str.RemovePrefix(prefix, notFoundValue: str);

		/// <returns>
		/// <paramref name="str"/> with the leading substring <paramref name="prefix"/> removed, or
		/// <paramref name="notFoundValue"/> if <paramref name="str"/> does not start with <paramref name="prefix"/>
		/// </returns>
		public static string RemovePrefix(this string str, string prefix, string notFoundValue) => str.StartsWith(prefix) ? str.Substring(prefix.Length, str.Length - prefix.Length) : notFoundValue;

		/// <returns>
		/// <paramref name="str"/> with the last char removed, or
		/// the original <paramref name="str"/> if the last char of <paramref name="str"/> is not <paramref name="suffix"/>
		/// </returns>
		public static string RemoveSuffix(this string str, char suffix) =>
			str.Length != 0 && str[str.Length - 1] == suffix
				? str.Substring(0, str.Length - 1)
				: str;

		/// <returns>
		/// <paramref name="str"/> with the trailing substring <paramref name="suffix"/> removed, or
		/// the original <paramref name="str"/> if <paramref name="str"/> does not end with <paramref name="suffix"/>
		/// </returns>
		public static string RemoveSuffix(this string str, string suffix) => str.RemoveSuffix(suffix, notFoundValue: str);

		/// <returns>
		/// <paramref name="str"/> with the trailing substring <paramref name="suffix"/> removed, or
		/// <paramref name="notFoundValue"/> if <paramref name="str"/> does not end with <paramref name="suffix"/>
		/// </returns>
		public static string RemoveSuffix(this string str, string suffix, string notFoundValue) => str.EndsWith(suffix) ? str.Substring(0, str.Length - suffix.Length) : notFoundValue;

		/// <returns>
		/// the substring of <paramref name="str"/> after the first occurrence of <paramref name="delimiter"/>, or
		/// the original <paramref name="str"/> if not found
		/// </returns>
		public static string SubstringAfter(this string str, string delimiter) => str.SubstringAfter(delimiter, notFoundValue: str);

		/// <returns>
		/// the substring of <paramref name="str"/> after the first occurrence of <paramref name="delimiter"/>, or
		/// <paramref name="notFoundValue"/> if not found
		/// </returns>
		public static string SubstringAfter(this string str, string delimiter, string notFoundValue)
		{
			var index = str.IndexOf(delimiter);
			return index < 0 ? notFoundValue : str.Substring(index + delimiter.Length, str.Length - index - delimiter.Length);
		}

		/// <returns>
		/// the substring of <paramref name="str"/> after the last occurrence of <paramref name="delimiter"/>, or
		/// the original <paramref name="str"/> if not found
		/// </returns>
		public static string SubstringAfterLast(this string str, char delimiter)
			=> str.SubstringAfterLast(delimiter, notFoundValue: str);

		/// <returns>
		/// the substring of <paramref name="str"/> after the last occurrence of <paramref name="delimiter"/>, or
		/// <paramref name="notFoundValue"/> if not found
		/// </returns>
		public static string SubstringAfterLast(this string str, char delimiter, string notFoundValue)
		{
			var index = str.LastIndexOf(delimiter);
			return index < 0 ? notFoundValue : str.Substring(index + 1, str.Length - index - 1);
		}

		/// <returns>
		/// the substring of <paramref name="str"/> before the first occurrence of <paramref name="delimiter"/>, or
		/// the original <paramref name="str"/> if not found
		/// </returns>
		public static string SubstringBefore(this string str, char delimiter) => str.SubstringBefore(delimiter, notFoundValue: str);

		/// <returns>
		/// the substring of <paramref name="str"/> before the first occurrence of <paramref name="delimiter"/>, or
		/// <paramref name="notFoundValue"/> if not found
		/// </returns>
		public static string SubstringBefore(this string str, char delimiter, string notFoundValue)
		{
			var index = str.IndexOf(delimiter);
			return index < 0 ? notFoundValue : str.Substring(0, index);
		}

		/// <returns>
		/// the substring of <paramref name="str"/> before the last occurrence of <paramref name="delimiter"/>, or
		/// the original <paramref name="str"/> if not found
		/// </returns>
		public static string SubstringBeforeLast(this string str, char delimiter) => str.SubstringBeforeLast(delimiter, notFoundValue: str);

		/// <returns>
		/// the substring of <paramref name="str"/> before the last occurrence of <paramref name="delimiter"/>, or
		/// <paramref name="notFoundValue"/> if not found
		/// </returns>
		public static string SubstringBeforeLast(this string str, char delimiter, string notFoundValue)
		{
			var index = str.LastIndexOf(delimiter);
			return index < 0 ? notFoundValue : str.Substring(0, index);
		}

		/// <returns>
		/// the substring of <paramref name="str"/> before the first occurrence of <paramref name="delimiter"/>, or
		/// <see langword="null"/> if not found
		/// </returns>
		public static string? SubstringBeforeOrNull(this string str, string delimiter)
		{
			var index = str.IndexOf(delimiter);
			return index < 0 ? null : str.Substring(0, index);
		}

		public static byte[] ToCharCodepointArray(this string str)
		{
			var a = new byte[str.Length];
			for (var i = 0; i < str.Length; i++) a[i] = (byte) char.ConvertToUtf32(str, i);
			return a;
		}

		/// <summary>
		/// splits a given <paramref name="str"/> by <paramref name="delimiter"/>,
		/// applies <paramref name="transform"/> to each part, then rejoins them
		/// </summary>
		/// <remarks><c>"abc,def,ghi".TransformFields(',', s => s.Reverse()) == "cba,fed,ihg"</c></remarks>
		public static string TransformFields(this string str, char delimiter, Func<string, string> transform)
			=> string.Join(delimiter.ToString(), str.Split(delimiter).Select(transform));
	}
}
