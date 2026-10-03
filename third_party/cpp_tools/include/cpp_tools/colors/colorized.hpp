// == file : colorized.hpp
// == author : Pierre Esterie
#ifndef CPP_TOOLS_COLORS_COLORIZED_HPP
#define CPP_TOOLS_COLORS_COLORIZED_HPP

// ESC[ 38;5;<n> m Select foreground color
// ESC[ 48;5;<n> m Select background color
//   0-  7:  standard colors (as in ESC [ 30–37 m)
//   8- 15:  high intensity colors (as in ESC [ 90–97 m)
//  16-231:  6 × 6 × 6 cube (216 colors): 16 + 36 × r + 6 × g + b (0 ≤ r, g, b ≤ 5)
// 232-255:  grayscale from black to white in 24 steps

#include <ostream>
#include <string>

namespace cpp_tools::colors
{
    inline constexpr auto get_color(const int R, const int G, const int B) -> int
    {
        constexpr auto sixteen{16};
        constexpr auto six{6};
        return (sixteen + six * six * R + six * G + B);
    }

    template<const int R, const int G, const int B>
    struct are_less_than_6
    {
        static constexpr auto six{6};
        using type = std::enable_if_t < R < six&& G < six&& B<six>;
    };

    template<const int R, const int G, const int B, bool Background = false, typename Enable = void>
    struct color
    {
        static constexpr auto six{6};
        static_assert(R < six && G < six && B < six, "R,G & B must be < 6 !");
    };

    template<const int R, const int G, const int B, bool Background>
    struct color<R, G, B, Background, typename are_less_than_6<R, G, B>::type> : std::string
    {
        constexpr color()
          : std::string("\033[38;5;" + std::to_string(get_color(R, G, B)) + "m")
        {
        }
    };

    template<const int R, const int G, const int B>
    struct color<R, G, B, true, typename are_less_than_6<R, G, B>::type> : std::string
    {
        constexpr color()
          : std::string("\033[48;5;" + std::to_string(get_color(R, G, B)) + "m")
        {
        }
    };

    template<const int R, const int G, const int B>
    using on_color = color<R, G, B, true>;

    template<const int Code>
    struct code : std::string
    {
        constexpr code()
          : std::string("\033[" + std::to_string(Code) + "m")
        {
        }
    };

    constexpr int black_{30};
    constexpr int red_{31};
    constexpr int green_{32};
    constexpr int yellow_{33};
    constexpr int blue_{34};
    constexpr int magenta_{35};
    constexpr int cyan_{36};
    constexpr int white_{37};
    constexpr int on_black_{41};
    constexpr int on_red_{41};
    constexpr int on_green_{42};
    constexpr int on_yellow_{43};
    constexpr int on_blue_{44};
    constexpr int on_magenta_{45};
    constexpr int on_cyan_{46};
    constexpr int on_white_{47};
    constexpr int reset_{0};
    constexpr int bold_{1};
    constexpr int faint_{2};
    constexpr int italic_{3};
    constexpr int underline_{4};
    constexpr int underline_off_{24};
    constexpr int slow_blink_{5};
    constexpr int rapid_blink_{6};
    constexpr int blink_off_{25};
    constexpr int reverse_{7};
    constexpr int overlined_{53};

    static auto black = code<black_>{};
    static auto red = code<red_>{};
    static auto green = code<green_>{};
    static auto yellow = code<yellow_>{};
    static auto blue = code<blue_>{};
    static auto magenta = code<magenta_>{};
    static auto cyan = code<cyan_>{};
    static auto white = code<white_>{};
    static auto on_black = code<on_black_>{};
    static auto on_red = code<on_red_>{};
    static auto on_green = code<on_green_>{};
    static auto on_yellow = code<on_yellow_>{};
    static auto on_blue = code<on_blue_>{};
    static auto on_magenta = code<on_magenta_>{};
    static auto on_cyan = code<on_cyan_>{};
    static auto on_white = code<on_white_>{};
    static auto reset = code<reset_>{};
    static auto bold = code<bold_>{};
    static auto faint = code<faint_>{};
    static auto italic = code<italic_>{};
    static auto underline = code<underline_>{};
    static auto underline_off = code<underline_off_>{};
    static auto slow_blink = code<slow_blink_>{};
    static auto rapid_blink = code<rapid_blink_>{};
    static auto blink_off = code<blink_off_>{};
    static auto reverse = code<reverse_>{};
    static auto overlined = code<overlined_>{};
}   // namespace scalfmm::colors

#endif
