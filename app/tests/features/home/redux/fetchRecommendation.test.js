import configureMockStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import nock from 'nock';

import {
  HOME_FETCH_RECOMMENDATION_BEGIN,
  HOME_FETCH_RECOMMENDATION_SUCCESS,
  HOME_FETCH_RECOMMENDATION_FAILURE,
  HOME_FETCH_RECOMMENDATION_DISMISS_ERROR,
} from '../../../../src/features/home/redux/constants';

import {
  fetchRecommendation,
  dismissFetchRecommendationError,
  reducer,
} from '../../../../src/features/home/redux/fetchRecommendation';

const middlewares = [thunk];
const mockStore = configureMockStore(middlewares);

describe('home/redux/fetchRecommendation', () => {
  afterEach(() => {
    nock.cleanAll();
  });

  it('dispatches success action when fetchRecommendation succeeds', () => {
    const store = mockStore({});

    return store.dispatch(fetchRecommendation())
      .then(() => {
        const actions = store.getActions();
        expect(actions[0]).toHaveProperty('type', HOME_FETCH_RECOMMENDATION_BEGIN);
        expect(actions[1]).toHaveProperty('type', HOME_FETCH_RECOMMENDATION_SUCCESS);
      });
  });

  it('dispatches failure action when fetchRecommendation fails', () => {
    const store = mockStore({});

    return store.dispatch(fetchRecommendation({ error: true }))
      .catch(() => {
        const actions = store.getActions();
        expect(actions[0]).toHaveProperty('type', HOME_FETCH_RECOMMENDATION_BEGIN);
        expect(actions[1]).toHaveProperty('type', HOME_FETCH_RECOMMENDATION_FAILURE);
        expect(actions[1]).toHaveProperty('data.error', expect.anything());
      });
  });

  it('returns correct action by dismissFetchRecommendationError', () => {
    const expectedAction = {
      type: HOME_FETCH_RECOMMENDATION_DISMISS_ERROR,
    };
    expect(dismissFetchRecommendationError()).toEqual(expectedAction);
  });

  it('handles action type HOME_FETCH_RECOMMENDATION_BEGIN correctly', () => {
    const prevState = { fetchRecommendationPending: false };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_RECOMMENDATION_BEGIN }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchRecommendationPending).toBe(true);
  });

  it('handles action type HOME_FETCH_RECOMMENDATION_SUCCESS correctly', () => {
    const prevState = { fetchRecommendationPending: true };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_RECOMMENDATION_SUCCESS, data: {} }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchRecommendationPending).toBe(false);
  });

  it('handles action type HOME_FETCH_RECOMMENDATION_FAILURE correctly', () => {
    const prevState = { fetchRecommendationPending: true };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_RECOMMENDATION_FAILURE, data: { error: new Error('some error') } }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchRecommendationPending).toBe(false);
    expect(state.fetchRecommendationError).toEqual(expect.anything());
  });

  it('handles action type HOME_FETCH_RECOMMENDATION_DISMISS_ERROR correctly', () => {
    const prevState = { fetchRecommendationError: new Error('some error') };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_RECOMMENDATION_DISMISS_ERROR }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchRecommendationError).toBe(null);
  });
});

