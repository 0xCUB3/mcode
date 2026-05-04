#pragma once
#include <cstdint>

namespace boost {
namespace date_time {
enum weekdays {
    Sunday = 0,
    Monday = 1,
    Tuesday = 2,
    Wednesday = 3,
    Thursday = 4,
    Friday = 5,
    Saturday = 6
};
}
namespace gregorian {

enum months_of_year { Jan = 1, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec };
using month = int;
using day = int;

class days {
public:
    explicit days(int n) : n_(n) {}
    int number_of_days() const { return n_; }
private:
    int n_;
};

class greg_weekday {
public:
    greg_weekday(int value = 0) : value_(value) {}
    int as_number() const { return value_; }
    operator int() const { return value_; }
private:
    int value_;
};

class date {
public:
    struct date_type {
        using month_type = int;
    };
    using year_type = int;
    using month_type = int;
    using day_type = int;

    date() : year_(0), month_(0), day_(0) {}
    date(year_type year, month_type month, day_type day)
        : year_(year), month_(month), day_(day) {}

    year_type year() const { return year_; }
    month_type month() const { return month_; }
    day_type day() const { return day_; }
    greg_weekday day_of_week() const {
        return greg_weekday(weekday_number(year_, month_, day_));
    }

    friend bool operator==(const date& lhs, const date& rhs) {
        return lhs.year_ == rhs.year_ && lhs.month_ == rhs.month_ && lhs.day_ == rhs.day_;
    }
    friend bool operator!=(const date& lhs, const date& rhs) { return !(lhs == rhs); }
    friend date operator+(const date& value, const days& delta) {
        return value.add_days(delta.number_of_days());
    }
    friend date operator-(const date& value, const days& delta) {
        return value.add_days(-delta.number_of_days());
    }

private:
    year_type year_;
    month_type month_;
    day_type day_;

    static bool leap(int year) {
        return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    }
    static int end_day(int year, int month) {
        static const int days_in_month[] = {0,31,28,31,30,31,30,31,31,30,31,30,31};
        return month == 2 && leap(year) ? 29 : days_in_month[month];
    }
    static int weekday_number(int year, int month, int day) {
        if (month < 3) {
            month += 12;
            --year;
        }
        int k = year % 100;
        int j = year / 100;
        int h = (day + (13 * (month + 1)) / 5 + k + k / 4 + j / 4 + 5 * j) % 7;
        return (h + 6) % 7;
    }
    date add_days(int count) const {
        int y = year_, m = month_, d = day_ + count;
        while (d > end_day(y, m)) {
            d -= end_day(y, m);
            if (++m > 12) {
                m = 1;
                ++y;
            }
        }
        while (d < 1) {
            if (--m < 1) {
                m = 12;
                --y;
            }
            d += end_day(y, m);
        }
        return date(y, m, d);
    }
};

inline int end_of_month_day(int year, int month) {
    static const int days_in_month[] = {0,31,28,31,30,31,30,31,31,30,31,30,31};
    bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    return month == 2 && leap ? 29 : days_in_month[month];
}

inline int days_in_month(int year, int month) {
    return end_of_month_day(year, month);
}

class first_day_of_the_week_after {
public:
    explicit first_day_of_the_week_after(date_time::weekdays day) : day_(day) {}
    date get_date(date start) const {
        date current = start + days(1);
        while (current.day_of_week().as_number() != static_cast<int>(day_)) {
            current = current + days(1);
        }
        return current;
    }
private:
    date_time::weekdays day_;
};

class first_day_of_the_week_in_month {
public:
    first_day_of_the_week_in_month(date_time::weekdays day, int month)
        : day_(day), month_(month) {}
    date get_date(int year) const { return nth_date(year, month_, day_, 1); }
private:
    date_time::weekdays day_;
    int month_;
    static date nth_date(int year, int month, date_time::weekdays day, int nth) {
        date current(year, month, 1);
        while (current.day_of_week().as_number() != static_cast<int>(day)) {
            current = current + days(1);
        }
        return current + days(7 * (nth - 1));
    }
};

class nth_day_of_the_week_in_month {
public:
    enum week_num { first = 1, second = 2, third = 3, fourth = 4 };
    nth_day_of_the_week_in_month(week_num nth, date_time::weekdays day, int month)
        : nth_(nth), day_(day), month_(month) {}
    date get_date(int year) const {
        date current(year, month_, 1);
        while (current.day_of_week().as_number() != static_cast<int>(day_)) {
            current = current + days(1);
        }
        return current + days(7 * (static_cast<int>(nth_) - 1));
    }
private:
    week_num nth_;
    date_time::weekdays day_;
    int month_;
};

class last_day_of_the_week_in_month {
public:
    last_day_of_the_week_in_month(date_time::weekdays day, int month)
        : day_(day), month_(month) {}
    date get_date(int year) const {
        date current(year, month_, end_of_month_day(year, month_));
        while (current.day_of_week().as_number() != static_cast<int>(day_)) {
            current = current - days(1);
        }
        return current;
    }
private:
    date_time::weekdays day_;
    int month_;
};

} // namespace gregorian
} // namespace boost
