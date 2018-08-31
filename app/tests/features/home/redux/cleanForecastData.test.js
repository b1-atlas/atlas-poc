import {
  HOME_CLEAN_FORECAST_DATA,
} from '../../../../src/features/home/redux/constants';

import {
  cleanForecastData,
  reducer,
} from '../../../../src/features/home/redux/cleanForecastData';

describe('home/redux/cleanForecastData', () => {
  it('returns correct action by cleanForecastData', () => {
    expect(cleanForecastData()).toHaveProperty('type', HOME_CLEAN_FORECAST_DATA);
  });

  it('handles action type HOME_CLEAN_FORECAST_DATA correctly', () => {
    const prevState = {};
    const state = reducer(
      prevState,
      { type: HOME_CLEAN_FORECAST_DATA }
    );
    // Should be immutable
    expect(state).not.toBe(prevState);

    // TODO: use real case expected value instead of {}.
    const expectedState = {};
    expect(state).toEqual(expectedState);
  });
});
