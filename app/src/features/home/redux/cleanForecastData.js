// Rekit uses a new approach to organizing actions and reducers. That is
// putting related actions and reducers in one file. See more at:
// https://medium.com/@nate_wang/a-new-approach-for-managing-redux-actions-91c26ce8b5da

import {
  HOME_CLEAN_FORECAST_DATA,
} from './constants';

export function cleanForecastData() {
  return {
    type: HOME_CLEAN_FORECAST_DATA,
  };
}

export function reducer(state, action) {
  switch (action.type) {
    case HOME_CLEAN_FORECAST_DATA:
      return {
        ...state,
        forecastPriceList: []
      };

    default:
      return state;
  }
}
