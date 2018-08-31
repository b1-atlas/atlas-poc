import initialState from './initialState';
import { reducer as fetchPurchasePriceListReducer } from './fetchPurchasePriceList';
import { reducer as fetchForecastPriceListReducer } from './fetchForecastPriceList';
import { reducer as cleanForecastDataReducer } from './cleanForecastData';
import { reducer as fetchRecommendationReducer } from './fetchRecommendation';

const reducers = [
  fetchPurchasePriceListReducer,
  fetchForecastPriceListReducer,
  cleanForecastDataReducer,
  fetchRecommendationReducer,
];

export default function reducer(state = initialState, action) {
  let newState;
  switch (action.type) {
    // Handle cross-topic actions here
    default:
      newState = state;
      break;
  }
  /* istanbul ignore next */
  return reducers.reduce((s, r) => r(s, action), newState);
}
