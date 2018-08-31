import configureMockStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import nock from 'nock';

import {
  HOME_FETCH_FORECAST_PRICE_LIST_BEGIN,
  HOME_FETCH_FORECAST_PRICE_LIST_SUCCESS,
  HOME_FETCH_FORECAST_PRICE_LIST_FAILURE,
  HOME_FETCH_FORECAST_PRICE_LIST_DISMISS_ERROR,
} from '../../../../src/features/home/redux/constants';

import {
  fetchForecastPriceList,
  dismissFetchForecastPriceListError,
  reducer,
} from '../../../../src/features/home/redux/fetchForecastPriceList';

const middlewares = [thunk];
const mockStore = configureMockStore(middlewares);

describe('home/redux/fetchForecastPriceList', () => {
  afterEach(() => {
    nock.cleanAll();
  });

  it('dispatches success action when fetchForecastPriceList succeeds', () => {
    const store = mockStore({});

    return store.dispatch(fetchForecastPriceList())
      .then(() => {
        const actions = store.getActions();
        expect(actions[0]).toHaveProperty('type', HOME_FETCH_FORECAST_PRICE_LIST_BEGIN);
        expect(actions[1]).toHaveProperty('type', HOME_FETCH_FORECAST_PRICE_LIST_SUCCESS);
      });
  });

  it('dispatches failure action when fetchForecastPriceList fails', () => {
    const store = mockStore({});

    return store.dispatch(fetchForecastPriceList({ error: true }))
      .catch(() => {
        const actions = store.getActions();
        expect(actions[0]).toHaveProperty('type', HOME_FETCH_FORECAST_PRICE_LIST_BEGIN);
        expect(actions[1]).toHaveProperty('type', HOME_FETCH_FORECAST_PRICE_LIST_FAILURE);
        expect(actions[1]).toHaveProperty('data.error', expect.anything());
      });
  });

  it('returns correct action by dismissFetchForecastPriceListError', () => {
    const expectedAction = {
      type: HOME_FETCH_FORECAST_PRICE_LIST_DISMISS_ERROR,
    };
    expect(dismissFetchForecastPriceListError()).toEqual(expectedAction);
  });

  it('handles action type HOME_FETCH_FORECAST_PRICE_LIST_BEGIN correctly', () => {
    const prevState = { fetchForecastPriceListPending: false };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_FORECAST_PRICE_LIST_BEGIN }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchForecastPriceListPending).toBe(true);
  });

  it('handles action type HOME_FETCH_FORECAST_PRICE_LIST_SUCCESS correctly', () => {
    const prevState = { fetchForecastPriceListPending: true };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_FORECAST_PRICE_LIST_SUCCESS, data: {} }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchForecastPriceListPending).toBe(false);
  });

  it('handles action type HOME_FETCH_FORECAST_PRICE_LIST_FAILURE correctly', () => {
    const prevState = { fetchForecastPriceListPending: true };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_FORECAST_PRICE_LIST_FAILURE, data: { error: new Error('some error') } }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchForecastPriceListPending).toBe(false);
    expect(state.fetchForecastPriceListError).toEqual(expect.anything());
  });

  it('handles action type HOME_FETCH_FORECAST_PRICE_LIST_DISMISS_ERROR correctly', () => {
    const prevState = { fetchForecastPriceListError: new Error('some error') };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_FORECAST_PRICE_LIST_DISMISS_ERROR }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchForecastPriceListError).toBe(null);
  });
});

