#include <range/v3/all.hpp>
namespace ranges {
template<class Rng>
class slide_view : public view_adaptor<slide_view<Rng>, Rng>
{
    CONCEPT_ASSERT(ForwardRange<Rng>());
    ranges::range_difference_type_t<Rng> step_;
    ranges::range_difference_type_t<Rng> size_;
    friend range_access;

    class adaptor : public adaptor_base
    {
        ranges::range_difference_type_t<Rng> step_;
        ranges::range_difference_type_t<Rng> size_;
        sentinel_t<Rng> end_;

    public:
        adaptor() = default;
        adaptor(ranges::range_difference_type_t<Rng> step,
                ranges::range_difference_type_t<Rng> size, sentinel_t<Rng> end)
          : step_(step)
          , size_(size)
          , end_(end)
        {}
        auto read(iterator_t<Rng> it) const
        {
            return view::take(make_iterator_range(std::move(it), end_), size_);
        }
        void next(iterator_t<Rng> &it)
        {
          // check that size_ many elements are left:
          // if (ranges::distance(it + step_, end_) >= size_) {
            ranges::advance(it, step_, end_);
          // otherwise: move to end_
        }
        void prev() = delete;
        void distance_to() = delete;
    };

    adaptor begin_adaptor()
    {
        return adaptor{step_, size_,ranges::end(this->base())};
    }

public:
    slide_view() = default;
    slide_view(Rng rng, ranges::range_difference_type_t<Rng> step, ranges::range_difference_type_t<Rng> size)
      : slide_view::view_adaptor(std::move(rng))
      , step_(step), size_(size)
    {}
};

auto
slide(std::size_t step, std::size_t size)
{
    return make_pipeable([=](auto &&rng) {
        using Rng = decltype(rng);
        return slide_view<view::all_t<Rng>>{
            view::all(std::forward<Rng>(rng)),
            static_cast<ranges::range_difference_type_t<Rng>>(step),
            static_cast<ranges::range_difference_type_t<Rng>>(size)};
    });
}
}
